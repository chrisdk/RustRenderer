//! GPU state: device, queue, pipelines, and scene/frame buffers.
//!
//! [`Renderer`] owns every piece of GPU state needed to trace a frame.
//! It is constructed once at startup (async, because WebGPU adapter and
//! device acquisition go through promise-based browser APIs), then reused
//! across every frame.
//!
//! ## Lifecycle
//!
//! ```text
//! Renderer::new()           — pick adapter → open device + queue
//!     ↓
//! renderer.upload_scene()   — BVH + materials → GPU storage buffers
//!     ↓
//! renderer.upload_camera()  — camera uniform → GPU uniform buffer
//!     ↓
//! renderer.render_frame()   — dispatch compute shader → writes output_buf
//!     ↓
//! renderer.get_pixels()     — copy output_buf → staging → Vec<u8>
//! ```
//!
//! ## Buffer / binding layout
//!
//! | Group | Binding | Name                | Type    | Usage              |
//! |-------|---------|---------------------|---------|---------------------|
//! | 0     | 0       | bvh_nodes           | storage | read               |
//! | 0     | 1       | bvh_tris            | storage | read               |
//! | 0     | 2       | materials           | storage | read               |
//! | 0     | 3       | tex_data            | storage | read               |
//! | 0     | 4       | tex_info            | storage | read               |
//! | 0     | 5       | env_pixels          | storage | read               |
//! | 0     | 6       | env_marginal_cdf    | storage | read               |
//! | 0     | 7       | env_conditional_cdf | storage | read               |
//! | 1     | 0       | camera              | uniform | read               |
//! | 1     | 1       | frame               | uniform | read               |
//! | 1     | 2       | output              | storage | read_write         |

use crate::accel::bvh::Bvh;
use crate::camera::CameraUniform;
use crate::scene::material::Material;
use crate::scene::texture::Texture;

// ============================================================================
// Supporting types
// ============================================================================

/// Per-frame shader constants: output dimensions, sample index, render flags,
/// environment map info, and user-adjustable lighting controls.
///
/// 56 bytes (one 32-byte block + one 24-byte extension), matching the WGSL
/// `FrameUniforms` struct exactly. Every field is 4-byte aligned and there is
/// no implicit padding — `bytemuck::Pod` is safe to derive.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrameUniforms {
    pub width:        u32,
    pub height:       u32,
    pub sample_index: u32,
    /// Bit flags passed to the shader.
    /// Bit 0: preview mode — cap bounce count to 2 for fast interactive rendering.
    /// Bit 1: IBL disabled — use procedural sky even if an env map is loaded.
    pub flags:        u32,
    /// Width of the loaded environment map in pixels; 0 when none is loaded.
    /// The shader treats env_width == 0 as "no env map, use procedural sky".
    pub env_width:    u32,
    /// Height of the loaded environment map in pixels; 0 when none is loaded.
    pub env_height:   u32,
    /// 1 = show the environment map (or procedural sky) as the scene background;
    /// 0 = use a neutral dark grey for rays that escape the scene.
    /// When 0, the env map can still light the scene via env NEE (explicit IBL
    /// shadow rays at each surface hit) — this toggle only affects what escaped
    /// rays contribute to the background, not whether IBL lights the scene.
    pub env_background: u32,
    /// Pre-computed constant for MIS PDF evaluation.
    ///
    /// `env_sample_weight = W * H / (total_sin_weighted_luminance * 2 * π²)`
    ///
    /// The PDF of a direction in solid-angle measure when importance-sampling
    /// the equirectangular env map equals `luminance(pixel) * env_sample_weight`.
    /// Set to 0.0 when no env map is loaded (disables MIS in the shader).
    pub env_sample_weight: f32,

    // ── User-adjustable lighting controls (offsets 32–55) ───────────────────

    /// Maximum path length in bounces (2–16). Clamped in `set_max_bounces`.
    /// Preview mode overrides this with `PREVIEW_BOUNCES` regardless.
    pub max_bounces:   u32,
    /// Sun horizontal angle in radians (0 = +Z axis, increases toward +X).
    /// Derived from the azimuth slider; the shader converts to a direction vector.
    pub sun_azimuth:   f32,
    /// Sun vertical angle above the horizon in radians (0 = horizon, π/2 = zenith).
    pub sun_elevation: f32,
    /// Multiplier for the analytical sun NEE radiance. 0.0 = invisible sun,
    /// 1.0 = physical default, 5.0 = blazing noon. Does not affect IBL env maps.
    pub sun_intensity: f32,
    /// Linear multiplier for all env-map and sky contributions (background,
    /// indirect IBL, env NEE). Does not affect emissive surfaces or the sun.
    pub ibl_scale:     f32,
    /// Exposure compensation in EV stops. Applied as `linear * 2^exposure`
    /// before ACES tone-mapping — adjusts display brightness without touching
    /// the accumulation buffer, so it works live during progressive rendering.
    pub exposure:      f32,
}

// ============================================================================
// Renderer
// ============================================================================

/// All GPU state required to trace rays through a scene.
pub struct Renderer {
    /// The logical GPU device: allocates buffers and creates pipelines.
    ///
    /// `pub(crate)` rather than private because `lib.rs` takes a raw pointer
    /// to this field in `get_pixels` / `raster_get_pixels` to carry the
    /// reference across an async boundary. All other access should go through
    /// [`Renderer::device_queue`] instead of touching this field directly.
    pub(crate) device: wgpu::Device,
    /// The GPU command queue: submits encoded command buffers for execution.
    ///
    /// Same caveat as `device` above — prefer [`Renderer::device_queue`].
    pub(crate) queue: wgpu::Queue,

    // ── Scene data (uploaded once per loaded scene) ──────────────────────────
    bvh_nodes_buf: Option<wgpu::Buffer>,
    bvh_tris_buf:  Option<wgpu::Buffer>,
    materials_buf: Option<wgpu::Buffer>,
    /// Flat RGBA8 pixel data for all scene textures, packed as one u32 per
    /// pixel (R in low bits, A in high bits). `tex_info_buf` maps a texture
    /// index to an offset into this buffer.
    tex_data_buf:  Option<wgpu::Buffer>,
    /// Per-texture metadata: pixel offset into `tex_data_buf`, width, height.
    tex_info_buf:  Option<wgpu::Buffer>,

    // ── Environment map (always valid; 1-pixel black dummy when not loaded) ──
    /// Linear-light RGB32F environment map pixels as `vec4<f32>` (A unused).
    /// Contains a 1×1 black pixel before any env map is loaded; replaced by
    /// `upload_environment`. Always valid so the bind group layout is fixed.
    env_pixels_buf: wgpu::Buffer,
    /// Marginal CDF over rows of the sin-weighted luminance image.
    /// `env_marginal_cdf[y]` = P(row ≤ y) — used for importance sampling.
    /// Contains a 1-element dummy `[1.0]` when no env map is loaded.
    env_marginal_cdf_buf: wgpu::Buffer,
    /// Per-row conditional CDFs: `env_conditional_cdf[y*W + x]` = P(col ≤ x | row = y).
    /// Used for the second stage of 2-D env map importance sampling.
    /// Contains a 1-element dummy `[1.0]` when no env map is loaded.
    env_conditional_cdf_buf: wgpu::Buffer,
    /// Normalization constant for the MIS PDF: `W * H / (total_sin_weighted_lum * 2π²)`.
    /// Set to 0.0 when no env map is loaded. Passed to the shader as
    /// `FrameUniforms::env_sample_weight`; 0.0 disables env NEE in the shader.
    env_sample_weight: f32,
    /// Dimensions of the loaded env map; (0, 0) when using the dummy.
    /// Passed to the shader in `FrameUniforms`; zero width triggers the
    /// procedural sky fallback.
    env_dims: (u32, u32),
    /// When `false`, the shader ignores the env map and falls back to the
    /// procedural sky even if one is loaded. Toggled by `set_ibl_enabled`.
    /// Corresponds to bit 1 of `FrameUniforms::flags`.
    pub ibl_enabled: bool,
    /// When `true` (default), escaped rays sample the environment map or
    /// procedural sky as the scene background. When `false`, all background
    /// rays return a neutral dark grey — useful for studio lighting looks
    /// where you want IBL illumination without a distracting sky backdrop.
    /// Toggled by `set_env_background`. Mapped to `FrameUniforms::env_background`.
    pub env_background: bool,

    // ── User-adjustable lighting controls ────────────────────────────────────
    /// Maximum path bounces (2–16). Default 12. Clamped on write.
    pub max_bounces:   u32,
    /// Sun horizontal angle in **radians**. Default ≈ 63° (1.0996 rad).
    pub sun_azimuth:   f32,
    /// Sun vertical angle in **radians**. Default ≈ 66° (1.1519 rad).
    pub sun_elevation: f32,
    /// Multiplier for the analytical sun NEE radiance. Default 1.0.
    pub sun_intensity: f32,
    /// Linear multiplier for env-map / sky contributions. Default 1.0.
    pub ibl_scale:     f32,
    /// Display exposure in EV stops, applied before tone-mapping. Default 0.0.
    pub exposure:      f32,

    // ── Per-frame data (updated before each dispatch) ────────────────────────
    /// Camera parameters. `@group(1) @binding(0)`.
    camera_buf: Option<wgpu::Buffer>,
    /// Frame dimensions + sample index. `@group(1) @binding(1)`.
    frame_buf:  Option<wgpu::Buffer>,
    /// Output pixel buffer (RGBA8). `@group(1) @binding(2)`.
    output_buf: Option<wgpu::Buffer>,
    /// Float accumulation buffer (RGBA f32, one vec4 per pixel). `@group(1) @binding(3)`.
    /// Stores the running sum of all samples so the shader can compute a
    /// progressive average without reading back to the CPU between dispatches.
    accum_buf:  Option<wgpu::Buffer>,
    /// Dimensions of the current output/accum buffers; used to detect resizes.
    output_dims: (u32, u32),

    // ── Pipeline (built lazily on first render_frame) ────────────────────────
    scene_bgl: Option<wgpu::BindGroupLayout>,
    frame_bgl: Option<wgpu::BindGroupLayout>,
    pipeline:  Option<wgpu::ComputePipeline>,
}

// ============================================================================
// Construction
// ============================================================================

impl Renderer {
    /// Selects a GPU adapter and opens a logical device on it.
    ///
    /// This function is `async` because WebGPU exposes hardware selection
    /// through browser promises (`navigator.gpu.requestAdapter()` etc.),
    /// which surface in Rust/WASM as `Future`s. On native targets wgpu
    /// drives the async internally.
    ///
    /// # Errors
    ///
    /// Returns a human-readable error string if no suitable adapter is found
    /// or if device creation fails.
    pub async fn new() -> Result<Self, String> {
        // wgpu 28: Instance::new takes &InstanceDescriptor (borrowed).
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // wgpu 28: request_adapter returns Result<Adapter, E>, not Option<Adapter>.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("no suitable GPU adapter found: {e}"))?;

        let info = adapter.get_info();
        log::info!("GPU adapter: {} ({:?})", info.name, info.backend);

        // wgpu 28: request_device takes only &DeviceDescriptor (trace-path arg removed).
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("path tracer"),
                    ..Default::default()
                },
            )
            .await
            .map_err(|e| format!("GPU device creation failed: {e}"))?;

        // Dummy GPU buffers keep all binding slots valid before any real environment
        // is loaded. The CDF buffers need at least one element (WebGPU prohibits
        // zero-size storage buffers). The single 1.0 value is a legal CDF: it
        // represents "the only row/column is the last one", which is never reached
        // because `env_sample_weight == 0.0` disables env importance sampling in
        // the shader before the CDF buffers are ever indexed.
        use wgpu::util::DeviceExt;
        let env_pixels_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("env_pixels_dummy"),
            contents: bytemuck::cast_slice(&[0.0f32; 4]),  // one vec4<f32>(0,0,0,0)
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let env_marginal_cdf_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("env_marginal_cdf_dummy"),
            contents: bytemuck::cast_slice(&[1.0f32]),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let env_conditional_cdf_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("env_conditional_cdf_dummy"),
            contents: bytemuck::cast_slice(&[1.0f32]),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self {
            device,
            queue,
            bvh_nodes_buf:           None,
            bvh_tris_buf:            None,
            materials_buf:           None,
            tex_data_buf:            None,
            tex_info_buf:            None,
            env_pixels_buf,
            env_marginal_cdf_buf,
            env_conditional_cdf_buf,
            env_sample_weight:       0.0,
            env_dims:                (0, 0),
            ibl_enabled:             true,
            env_background:          true,
            // Lighting controls: defaults reproduce the old hardcoded shader values.
            // Sun angles chosen to match the original SUN_DIR = (0.3651, 0.9129, 0.1826):
            //   elevation = asin(0.9129) ≈ 65.6° → 66°
            //   azimuth   = atan2(0.3651, 0.1826) / cos(66°) ≈ 63°
            max_bounces:   12,
            sun_azimuth:   63.0_f32.to_radians(),
            sun_elevation: 66.0_f32.to_radians(),
            sun_intensity: 1.0,
            ibl_scale:     1.0,
            exposure:      0.0,
            camera_buf:     None,
            frame_buf:      None,
            output_buf:     None,
            accum_buf:      None,
            output_dims:    (0, 0),
            scene_bgl:      None,
            frame_bgl:      None,
            pipeline:       None,
        })
    }
}

// ============================================================================
// Device / queue access
// ============================================================================

impl Renderer {
    /// Borrows the GPU device and command queue.
    ///
    /// The [`RasterRenderer`] does not own a device/queue of its own — it
    /// borrows them from here at each call site. Use this accessor rather than
    /// accessing the fields directly so that the `pub(crate)` visibility on
    /// those fields remains an implementation detail rather than a stable API.
    ///
    /// [`RasterRenderer`]: super::RasterRenderer
    pub fn device_queue(&self) -> (&wgpu::Device, &wgpu::Queue) {
        (&self.device, &self.queue)
    }

    /// Borrows the environment-pixel storage buffer.
    ///
    /// Always valid — contains a 1×1 black pixel before any env map is loaded,
    /// replaced by [`upload_environment`]. The rasteriser binds this buffer
    /// directly so it never needs its own copy of the environment data.
    ///
    /// [`upload_environment`]: Self::upload_environment
    pub(crate) fn env_pixels_buf(&self) -> &wgpu::Buffer {
        &self.env_pixels_buf
    }

    /// Returns the dimensions `(width, height)` of the currently loaded
    /// environment map, or `(0, 0)` when using the dummy 1×1 black pixel.
    ///
    /// The rasteriser uses this to decide whether to sample the env map or
    /// fall back to the procedural sky gradient.
    pub(crate) fn env_dims(&self) -> (u32, u32) {
        self.env_dims
    }
}

// ============================================================================
// Scene and camera uploads
// ============================================================================

impl Renderer {
    /// Uploads BVH geometry, materials, and textures into GPU storage buffers.
    ///
    /// Calling this a second time drops the old buffers and creates fresh ones.
    pub fn upload_scene(&mut self, bvh: &Bvh, materials: &[Material], textures: &[Texture]) {
        use wgpu::util::DeviceExt;
        let flags = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        self.bvh_nodes_buf = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label:    Some("bvh_nodes"),
                contents: bytemuck::cast_slice(&bvh.nodes),
                usage:    flags,
            },
        ));
        self.bvh_tris_buf = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label:    Some("bvh_triangles"),
                contents: bytemuck::cast_slice(&bvh.triangles),
                usage:    flags,
            },
        ));
        self.materials_buf = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label:    Some("materials"),
                contents: bytemuck::cast_slice(materials),
                usage:    flags,
            },
        ));

        // Pack all textures into flat GPU storage buffers.
        // See renderer::pack_textures for the encoding details and zero-size guard.
        let (tex_data, tex_info) = super::pack_textures(textures);

        self.tex_data_buf = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label:    Some("tex_data"),
                contents: bytemuck::cast_slice(&tex_data),
                usage:    flags,
            },
        ));
        self.tex_info_buf = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label:    Some("tex_info"),
                contents: bytemuck::cast_slice(&tex_info),
                usage:    flags,
            },
        ));

        log::info!(
            "scene uploaded: {} BVH nodes, {} triangles, {} materials, {} textures ({} MB pixel data)",
            bvh.nodes.len(), bvh.triangles.len(), materials.len(),
            textures.len(), tex_data.len() * 4 / 1_000_000,
        );
    }

    /// Uploads (or updates) the camera uniform buffer.
    ///
    /// On the first call the buffer is allocated and filled. On every subsequent
    /// call `queue.write_buffer` updates it in-place — no allocation, no stall.
    pub fn upload_camera(&mut self, uniform: &CameraUniform) {
        let data = bytemuck::bytes_of(uniform);
        match &self.camera_buf {
            Some(buf) => self.queue.write_buffer(buf, 0, data),
            None => {
                use wgpu::util::DeviceExt;
                self.camera_buf = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label:    Some("camera_uniform"),
                        contents: data,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    },
                ));
            }
        }
        log::debug!("camera uploaded: pos={:?}", uniform.position);
    }

    /// Uploads a decoded HDR environment map to the GPU.
    ///
    /// `pixels` must be `width * height * 4` f32 values in row-major order,
    /// laid out as `[R, G, B, 1.0]` per pixel (vec4 with padding). See
    /// `scene::environment::decode_hdr` for the producer.
    ///
    /// Also builds the marginal + conditional CDFs used by the shader for
    /// importance sampling and uploads them to their respective storage buffers.
    /// Setting `env_sample_weight > 0` enables env NEE and MIS in the shader.
    ///
    /// Can be called multiple times to swap environments; old buffers are
    /// dropped automatically.
    pub fn upload_environment(&mut self, width: u32, height: u32, pixels: Vec<f32>) {
        use wgpu::util::DeviceExt;

        // Build importance-sampling CDFs on the CPU before uploading.
        let (marginal_cdf, conditional_cdf, sample_weight) =
            build_env_cdfs(&pixels, width as usize, height as usize);

        self.env_pixels_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("env_pixels"),
            contents: bytemuck::cast_slice(&pixels),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.env_marginal_cdf_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("env_marginal_cdf"),
            contents: bytemuck::cast_slice(&marginal_cdf),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.env_conditional_cdf_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("env_conditional_cdf"),
            contents: bytemuck::cast_slice(&conditional_cdf),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.env_sample_weight = sample_weight;
        self.env_dims = (width, height);

        log::info!(
            "env map uploaded: {}×{} ({} MB), sample_weight={:.4}",
            width, height, pixels.len() * 4 / 1_000_000, sample_weight,
        );
    }

    /// Removes the loaded HDR environment map and reverts to the procedural
    /// sky gradient.
    ///
    /// Replaces all env buffers with single-element dummies and resets
    /// `env_dims` to `(0, 0)`. Setting `env_sample_weight = 0.0` disables env
    /// NEE in the shader so the dummy CDF buffers are never indexed.
    ///
    /// Calling this before a scene is loaded is safe and a no-op for rendering.
    pub fn unload_environment(&mut self) {
        use wgpu::util::DeviceExt;
        self.env_pixels_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("env_pixels_dummy"),
            contents: bytemuck::cast_slice(&[0.0f32; 4]),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.env_marginal_cdf_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("env_marginal_cdf_dummy"),
            contents: bytemuck::cast_slice(&[1.0f32]),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.env_conditional_cdf_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("env_conditional_cdf_dummy"),
            contents: bytemuck::cast_slice(&[1.0f32]),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.env_sample_weight = 0.0;
        self.env_dims = (0, 0);
        log::info!("environment unloaded; reverting to procedural sky");
    }

    /// Returns `true` once [`upload_scene`] has been called.
    pub fn is_scene_loaded(&self) -> bool {
        self.bvh_nodes_buf.is_some()
    }
}

// ============================================================================
// Environment importance sampling — CPU CDF build
// ============================================================================

/// Build 1-D CDFs for importance sampling an equirectangular HDR environment map.
///
/// Returns `(marginal_cdf, conditional_cdf, sample_weight)`:
/// - `marginal_cdf[y]`         — CDF over rows: `P(row ≤ y)`.
/// - `conditional_cdf[y*W+x]`  — CDF over columns within row y: `P(col ≤ x | row = y)`.
/// - `sample_weight`           — normalisation constant `W·H / (total_energy · 2π²)`.
///
/// The PDF for sampling direction ω that maps to pixel `(x, y)` is:
/// `pdf(ω) = luminance(x, y) × sample_weight`
///
/// Setting this equal to the BRDF PDF in the power heuristic gives unbiased
/// MIS weights for the two-strategy estimator (env NEE + BRDF bounce).
///
/// ## Why sin(θ) weighting?
///
/// Equirectangular images give each pixel equal (u, v) area, but the solid angle
/// subtended by a pixel shrinks near the poles as `sin(θ)`. Rows near the equator
/// cover more sky and should be sampled proportionally more.  Multiplying each
/// pixel's luminance by `sin(θ)` corrects for this — the resulting PDF is uniform
/// over equal solid-angle patches rather than equal pixel-area patches.
pub fn build_env_cdfs(pixels: &[f32], width: usize, height: usize) -> (Vec<f32>, Vec<f32>, f32) {
    use std::f32::consts::PI;

    // Compute sin-weighted luminance for every pixel.
    // The pixel layout is vec4<f32>: R, G, B, A — so stride = 4.
    let mut weighted_lum = vec![0.0f32; width * height];
    for y in 0..height {
        // Latitude of the pixel centre: θ = π·(y+0.5)/H maps y=0 → top, y=H-1 → bottom.
        let theta     = PI * (y as f32 + 0.5) / height as f32;
        let sin_theta = theta.sin();
        for x in 0..width {
            let i = (y * width + x) * 4;
            if i + 2 < pixels.len() {
                let lum = 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
                weighted_lum[y * width + x] = lum * sin_theta;
            }
        }
    }

    // Row energies: sum of weighted luminance in each row.
    // Floor at 1e-10 so zero-energy rows (pitch-black bands) still produce a
    // valid conditional CDF (avoids a 0/0 division).
    let mut row_energy = vec![0.0f32; height];
    for y in 0..height {
        row_energy[y] = weighted_lum[y * width..(y + 1) * width]
            .iter()
            .sum::<f32>()
            .max(1e-10);
    }

    let total_energy: f32 = row_energy.iter().sum::<f32>().max(1e-10);

    // ── Marginal CDF over rows ───────────────────────────────────────────────
    let mut marginal_cdf = vec![0.0f32; height];
    let mut cumsum = 0.0f32;
    for y in 0..height {
        cumsum += row_energy[y] / total_energy;
        marginal_cdf[y] = cumsum.min(1.0);
    }

    // ── Conditional CDFs over columns within each row ────────────────────────
    let mut conditional_cdf = vec![0.0f32; width * height];
    for y in 0..height {
        let mut cumsum = 0.0f32;
        for x in 0..width {
            cumsum += weighted_lum[y * width + x] / row_energy[y];
            conditional_cdf[y * width + x] = cumsum.min(1.0);
        }
    }

    // ── Sample weight ────────────────────────────────────────────────────────
    //
    // The PDF of pixel (x, y) is:
    //   pdf_pixel(x, y) = lum(x,y)·sin(θ_y) / total_energy
    //
    // The solid angle of one pixel is:
    //   dω(x, y) = (2π/W)·(π/H)·sin(θ_y)
    //
    // Dividing gives the PDF in solid-angle measure:
    //   pdf_ω = pdf_pixel / dω = lum(x,y)·W·H / (total_energy·2π²)
    //
    // Everything except `lum(x,y)` is constant across the image — that's the
    // sample_weight.  The shader evaluates the env-map PDF for any direction
    // as `luminance(env_sample(dir)) * frame.env_sample_weight`.
    let sample_weight = (width * height) as f32 / (total_energy * 2.0 * PI * PI);

    (marginal_cdf, conditional_cdf, sample_weight)
}

// ============================================================================
// Pipeline construction (lazy, called on first render_frame)
// ============================================================================

impl Renderer {
    fn build_pipeline(&mut self) {
        // ── Shader ──────────────────────────────────────────────────────────
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("trace.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/trace.wgsl").into()
            ),
        });

        // ── Bind group layouts ───────────────────────────────────────────────
        // Group 0: scene data — eight read-only storage buffers.
        let scene_bgl = self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("scene_bgl"),
                entries: &[
                    storage_ro(0), // bvh_nodes
                    storage_ro(1), // bvh_tris
                    storage_ro(2), // materials
                    storage_ro(3), // tex_data           (flat RGBA8 pixel data)
                    storage_ro(4), // tex_info           (per-texture offset/size)
                    storage_ro(5), // env_pixels         (HDR env map as vec4<f32>)
                    storage_ro(6), // env_marginal_cdf   (row importance weights)
                    storage_ro(7), // env_conditional_cdf (per-row column weights)
                ],
            }
        );

        // Group 1: per-frame data — two uniforms + two read-write storage buffers.
        let frame_bgl = self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("frame_bgl"),
                entries: &[
                    uniform(0),     // camera
                    uniform(1),     // frame (width, height, sample_index)
                    storage_rw(2),  // output pixel buffer (RGBA8 u32)
                    storage_rw(3),  // accumulation buffer (RGBA f32)
                ],
            }
        );

        // ── Pipeline layout and compute pipeline ─────────────────────────────
        let layout = self.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label:                Some("trace_layout"),
                bind_group_layouts:   &[&scene_bgl, &frame_bgl],
                // wgpu 28: push_constant_ranges removed; replaced by immediate_size.
                immediate_size: 0,
            }
        );

        let pipeline = self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label:       Some("trace_pipeline"),
                layout:      Some(&layout),
                module:      &shader,
                // wgpu 28: entry_point is Option<&str> again.
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache:       None,
            }
        );

        self.scene_bgl = Some(scene_bgl);
        self.frame_bgl = Some(frame_bgl);
        self.pipeline  = Some(pipeline);

        log::info!("compute pipeline ready");
    }
}

// ── Helpers for building bind group layout entries ───────────────────────────

fn storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    }
}

fn storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    }
}

fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    }
}

// ============================================================================
// Rendering
// ============================================================================

impl Renderer {
    /// Dispatches the compute shader for one sample pass.
    ///
    /// Builds the pipeline on the first call (shader compilation happens here).
    /// The output is written to an internal `output_buf`; call [`get_pixels`]
    /// afterwards to read the result back to the CPU.
    ///
    /// Panics if [`upload_scene`] or [`upload_camera`] has not been called.
    pub fn render_frame(&mut self, width: u32, height: u32, sample_index: u32, preview: bool) {
        assert!(self.is_scene_loaded(), "render_frame called before upload_scene");
        assert!(self.camera_buf.is_some(), "render_frame called before upload_camera");

        // Lazy pipeline build — compiles the WGSL shader on first call.
        if self.pipeline.is_none() {
            self.build_pipeline();
        }

        // (Re)create the output and accumulation buffers when dimensions change.
        let pixel_count = (width * height) as u64;
        let output_size = pixel_count * 4;   // 4 bytes per RGBA8 pixel
        let accum_size  = pixel_count * 16;  // 16 bytes per RGBA f32 pixel
        if self.output_dims != (width, height) {
            self.output_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("output"),
                size:               output_size,
                usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.accum_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("accum"),
                size:               accum_size,
                usage:              wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
            self.output_dims = (width, height);
        }

        // Upload frame uniforms.
        // Bit 0: preview mode (cap bounces for interactive speed).
        // Bit 1: IBL disabled (use procedural sky even if an env map is loaded).
        let flags = if preview          { 1u32 } else { 0u32 }
                  | if !self.ibl_enabled { 2u32 } else { 0u32 };
        let frame_data = FrameUniforms {
            width, height, sample_index, flags,
            env_width:         self.env_dims.0,
            env_height:        self.env_dims.1,
            env_background:    u32::from(self.env_background),
            // Zero when no env is loaded — disables env NEE + MIS in the shader.
            env_sample_weight: self.env_sample_weight,
            max_bounces:   self.max_bounces,
            sun_azimuth:   self.sun_azimuth,
            sun_elevation: self.sun_elevation,
            sun_intensity: self.sun_intensity,
            ibl_scale:     self.ibl_scale,
            exposure:      self.exposure,
        };
        match &self.frame_buf {
            Some(buf) => self.queue.write_buffer(buf, 0, bytemuck::bytes_of(&frame_data)),
            None => {
                use wgpu::util::DeviceExt;
                self.frame_buf = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label:    Some("frame_uniform"),
                        contents: bytemuck::bytes_of(&frame_data),
                        usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    },
                ));
            }
        }

        // Build bind groups (cheap descriptor objects; fine to recreate per frame).
        let scene_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("scene_bg"),
            layout:  self.scene_bgl.as_ref().expect("build_pipeline not called"),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bvh_nodes_buf.as_ref().expect("upload_scene not called").as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bvh_tris_buf.as_ref().expect("upload_scene not called").as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.materials_buf.as_ref().expect("upload_scene not called").as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.tex_data_buf.as_ref().expect("upload_scene not called").as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.tex_info_buf.as_ref().expect("upload_scene not called").as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.env_pixels_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.env_marginal_cdf_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.env_conditional_cdf_buf.as_entire_binding() },
            ],
        });

        let frame_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("frame_bg"),
            layout:  self.frame_bgl.as_ref().expect("build_pipeline not called"),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.camera_buf.as_ref().expect("upload_camera not called").as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.frame_buf.as_ref().expect("frame buffer missing").as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.output_buf.as_ref().expect("output buffer missing").as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.accum_buf.as_ref().expect("accum buffer missing").as_entire_binding() },
            ],
        });

        // Encode and submit the compute pass.
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("trace") }
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("trace_pass"), timestamp_writes: None }
            );
            pass.set_pipeline(self.pipeline.as_ref().expect("build_pipeline not called"));
            pass.set_bind_group(0, &scene_bg, &[]);
            pass.set_bind_group(1, &frame_bg, &[]);
            // Workgroup size is 8×8; dispatch enough groups to cover every pixel.
            pass.dispatch_workgroups(
                (width  + 7) / 8,
                (height + 7) / 8,
                1,
            );
        }
        self.queue.submit(Some(encoder.finish()));
        log::debug!("render_frame dispatched ({}×{})", width, height);
    }

    /// Copies the rendered pixels from the GPU and returns them as raw RGBA8 bytes.
    ///
    /// This is `async` because WebGPU buffer mapping goes through a browser
    /// Promise. On native, `device.poll(Wait)` blocks synchronously instead.
    ///
    /// The returned `Vec<u8>` is `width * height * 4` bytes, in row-major order,
    /// with R at byte 0 of each pixel (matching the `ImageData` layout expected
    /// by the canvas 2D API).
    pub async fn get_pixels(&self, width: u32, height: u32) -> Vec<u8> {
        let size = (width * height * 4) as u64;

        // Staging buffer: CPU-readable, receives a copy of the output.
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("pixel_staging"),
            size,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("readback") }
        );
        encoder.copy_buffer_to_buffer(
            self.output_buf.as_ref().expect("get_pixels called before render_frame"),
            0,
            &staging,
            0,
            size,
        );
        self.queue.submit(Some(encoder.finish()));

        // Map the staging buffer asynchronously.
        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).ok(); });

        // Drive completion: poll on native, yield to the JS event loop on WASM.
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        #[cfg(target_arch = "wasm32")]
        {
            // A single microtask yield lets the WebGPU Promise resolve and fire
            // the map_async callback before we try to receive on rx.
            let _ = wasm_bindgen_futures::JsFuture::from(
                js_sys::Promise::resolve(&wasm_bindgen::JsValue::UNDEFINED)
            ).await;
        }

        rx.await
            .expect("map_async channel closed")
            .expect("buffer mapping failed");

        let data   = slice.get_mapped_range();
        let pixels = data.to_vec();
        drop(data);
        staging.unmap();
        pixels
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    use std::mem::{offset_of, size_of};

    /// `FrameUniforms` is uploaded verbatim to a GPU uniform buffer every frame.
    /// The WGSL `FrameUniforms` struct must match byte-for-byte. If a field is
    /// added, removed, or reordered on either side without updating the other,
    /// the shader will silently read garbage for every frame constant — wrong
    /// output size, wrong sample index, wrong flags, broken IBL switch, or
    /// an always-visible/invisible environment background.
    #[test]
    fn test_frame_uniforms_gpu_layout() {
        assert_eq!(size_of::<FrameUniforms>(), 56,
            "FrameUniforms must be 56 bytes (32 original + 24 lighting controls)");
        assert_eq!(offset_of!(FrameUniforms, width),             0);
        assert_eq!(offset_of!(FrameUniforms, height),            4);
        assert_eq!(offset_of!(FrameUniforms, sample_index),      8);
        assert_eq!(offset_of!(FrameUniforms, flags),             12);
        assert_eq!(offset_of!(FrameUniforms, env_width),         16);
        assert_eq!(offset_of!(FrameUniforms, env_height),        20);
        assert_eq!(offset_of!(FrameUniforms, env_background),    24);
        assert_eq!(offset_of!(FrameUniforms, env_sample_weight), 28);
        assert_eq!(offset_of!(FrameUniforms, max_bounces),       32);
        assert_eq!(offset_of!(FrameUniforms, sun_azimuth),       36);
        assert_eq!(offset_of!(FrameUniforms, sun_elevation),     40);
        assert_eq!(offset_of!(FrameUniforms, sun_intensity),     44);
        assert_eq!(offset_of!(FrameUniforms, ibl_scale),         48);
        assert_eq!(offset_of!(FrameUniforms, exposure),          52);
        // Verify Pod is satisfied (bytemuck requires no padding bytes and no
        // invalid bit patterns — both f32 and u32 are Pod, so this is fine).
        let _: &[u8] = bytemuck::bytes_of(&FrameUniforms::zeroed());
    }

    /// A uniform-luminance env map has equal pixel weights within each row, so
    /// every row's conditional CDF should be exactly uniform: CDF[x] = (x+1)/W.
    ///
    /// The marginal CDF over rows is NOT uniform even for uniform luminance,
    /// because the equirectangular sin(θ) correction gives equatorial rows more
    /// weight than polar rows (they subtend more solid angle).  We verify the
    /// marginal CDF is consistent with the expected per-row sin(θ) weights.
    #[test]
    fn test_build_env_cdfs_uniform() {
        use std::f32::consts::PI;
        let w = 4usize; let h = 4usize;
        // Four f32s per pixel (vec4); luminance = 0.2126+0.7152+0.0722 = 1.0.
        let pixels = vec![1.0f32; w * h * 4];
        let (marginal, cond, _weight) = build_env_cdfs(&pixels, w, h);

        // ── Marginal CDF: verify against analytically computed sin-weighted totals ──
        // Row energy ∝ sin(θ_y) for uniform luminance.
        let sins: Vec<f32> = (0..h).map(|y| {
            let theta = PI * (y as f32 + 0.5) / h as f32;
            theta.sin()
        }).collect();
        let total: f32 = sins.iter().sum();
        let mut expected_marginal = Vec::new();
        let mut cumsum = 0.0f32;
        for &s in &sins { cumsum += s / total; expected_marginal.push(cumsum.min(1.0)); }

        assert_eq!(marginal.len(), h);
        for (y, (&got, &exp)) in marginal.iter().zip(expected_marginal.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-5,
                "marginal_cdf[{y}]: got {got:.6}, expected {exp:.6}");
        }

        // ── Conditional CDFs: uniform within each row ─────────────────────────
        assert_eq!(cond.len(), w * h);
        for y in 0..h {
            for x in 0..w {
                let expected = (x + 1) as f32 / w as f32;
                let got = cond[y * w + x];
                assert!((got - expected).abs() < 1e-5,
                    "cond_cdf[{y},{x}]: got {got:.6}, expected {expected:.6}");
            }
        }
    }

    /// A single bright pixel should push all CDF mass to its row and column.
    #[test]
    fn test_build_env_cdfs_single_bright_pixel() {
        let w = 4usize; let h = 4usize;
        let mut pixels = vec![0.0f32; w * h * 4];
        // Set pixel (col=2, row=1) to white: luminance = 1.0.
        let bright_row = 1usize; let bright_col = 2usize;
        let i = (bright_row * w + bright_col) * 4;
        pixels[i]     = 1.0; // R
        pixels[i + 1] = 1.0; // G
        pixels[i + 2] = 1.0; // B

        let (marginal, cond, sample_weight) = build_env_cdfs(&pixels, w, h);

        // Marginal CDF: all mass in row 1, so CDF is 0 for rows 0, then jumps to 1.
        assert!(marginal[0] < 1e-5, "row 0 should have near-zero mass");
        assert!((marginal[bright_row] - 1.0).abs() < 1e-5,
            "all mass should be at row {bright_row}");

        // Conditional CDF for bright_row: mass is entirely in bright_col.
        for x in 0..bright_col {
            assert!(cond[bright_row * w + x] < 1e-5,
                "cond_cdf[{bright_row},{x}] should be near zero before bright_col");
        }
        assert!((cond[bright_row * w + bright_col] - 1.0).abs() < 1e-5);

        // sample_weight must be positive.
        assert!(sample_weight > 0.0, "sample_weight should be positive");
    }

    /// CDFs must be non-decreasing and end exactly at 1.0.
    #[test]
    fn test_build_env_cdfs_monotone() {
        let w = 8usize; let h = 4usize;
        // Random-ish luminance pattern.
        let mut pixels = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let v = ((x + y * 3) % 5) as f32 / 5.0;
                let i = (y * w + x) * 4;
                pixels[i] = v; pixels[i + 1] = v; pixels[i + 2] = v;
            }
        }

        let (marginal, cond, _) = build_env_cdfs(&pixels, w, h);

        // Check marginal is non-decreasing and ends at 1.
        for y in 1..h { assert!(marginal[y] >= marginal[y - 1]); }
        assert!((marginal[h - 1] - 1.0).abs() < 1e-5);

        // Check each row's conditional CDF.
        for y in 0..h {
            for x in 1..w { assert!(cond[y * w + x] >= cond[y * w + x - 1]); }
            assert!((cond[y * w + w - 1] - 1.0).abs() < 1e-5);
        }
    }

    // GpuTexInfo layout is tested once, authoritatively, in renderer::tests.
}
