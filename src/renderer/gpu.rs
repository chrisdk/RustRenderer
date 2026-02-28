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
//! | Group | Binding | Name         | Type    | Usage              |
//! |-------|---------|--------------|---------|---------------------|
//! | 0     | 0       | bvh_nodes    | storage | read               |
//! | 0     | 1       | bvh_tris     | storage | read               |
//! | 0     | 2       | materials    | storage | read               |
//! | 0     | 3       | tex_data     | storage | read               |
//! | 0     | 4       | tex_info     | storage | read               |
//! | 1     | 0       | camera       | uniform | read               |
//! | 1     | 1       | frame        | uniform | read               |
//! | 1     | 2       | output       | storage | read_write         |

use crate::accel::bvh::Bvh;
use crate::camera::CameraUniform;
use crate::scene::material::Material;
use crate::scene::texture::Texture;

// ============================================================================
// Supporting types
// ============================================================================

/// Per-frame shader constants: output dimensions, sample index, render flags,
/// and environment map info.
///
/// 32 bytes (two 16-byte rows), matching the WGSL `FrameUniforms` struct exactly.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrameUniforms {
    pub width:        u32,
    pub height:       u32,
    pub sample_index: u32,
    /// Bit flags passed to the shader.
    /// Bit 0: preview mode — cap bounce count to 2 for fast interactive rendering.
    pub flags:        u32,
    /// Width of the loaded environment map in pixels; 0 when none is loaded.
    /// The shader treats env_width == 0 as "no env map, use procedural sky".
    pub env_width:    u32,
    /// Height of the loaded environment map in pixels; 0 when none is loaded.
    pub env_height:   u32,
    pub _pad0:        u32,
    pub _pad1:        u32,
}

/// Per-texture metadata uploaded to the GPU alongside the flat pixel data.
///
/// The shader uses this to locate each texture's pixel data within the shared
/// `tex_data` storage buffer and to convert UV coordinates to pixel addresses.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTextureInfo {
    /// Pixel index (not byte offset) into `tex_data` where this texture starts.
    offset: u32,
    width:  u32,
    height: u32,
    _pad:   u32,
}

// ============================================================================
// Renderer
// ============================================================================

/// All GPU state required to trace rays through a scene.
pub struct Renderer {
    /// The logical GPU device: allocates buffers and creates pipelines.
    pub(crate) device: wgpu::Device,
    /// The GPU command queue: submits encoded command buffers for execution.
    pub(crate) queue: wgpu::Queue,

    // ── Scene data (uploaded once per loaded scene) ──────────────────────────
    pub(crate) bvh_nodes_buf: Option<wgpu::Buffer>,
    pub(crate) bvh_tris_buf:  Option<wgpu::Buffer>,
    pub(crate) materials_buf: Option<wgpu::Buffer>,
    /// Flat RGBA8 pixel data for all scene textures, packed as one u32 per
    /// pixel (R in low bits, A in high bits). `tex_info_buf` maps a texture
    /// index to an offset into this buffer.
    pub(crate) tex_data_buf:  Option<wgpu::Buffer>,
    /// Per-texture metadata: pixel offset into `tex_data_buf`, width, height.
    pub(crate) tex_info_buf:  Option<wgpu::Buffer>,

    // ── Environment map (always valid; 1-pixel black dummy when not loaded) ──
    /// Linear-light RGB32F environment map pixels as `vec4<f32>` (A unused).
    /// Contains a 1×1 black pixel before any env map is loaded; replaced by
    /// `upload_environment`. Always valid so the bind group layout is fixed.
    pub(crate) env_pixels_buf: wgpu::Buffer,
    /// Dimensions of the loaded env map; (0, 0) when using the dummy.
    /// Passed to the shader in `FrameUniforms`; zero width triggers the
    /// procedural sky fallback.
    pub(crate) env_dims: (u32, u32),

    // ── Per-frame data (updated before each dispatch) ────────────────────────
    /// Camera parameters. `@group(1) @binding(0)`.
    pub(crate) camera_buf: Option<wgpu::Buffer>,
    /// Frame dimensions + sample index. `@group(1) @binding(1)`.
    pub(crate) frame_buf:  Option<wgpu::Buffer>,
    /// Output pixel buffer (RGBA8). `@group(1) @binding(2)`.
    pub(crate) output_buf: Option<wgpu::Buffer>,
    /// Float accumulation buffer (RGBA f32, one vec4 per pixel). `@group(1) @binding(3)`.
    /// Stores the running sum of all samples so the shader can compute a
    /// progressive average without reading back to the CPU between dispatches.
    pub(crate) accum_buf:  Option<wgpu::Buffer>,
    /// Dimensions of the current output/accum buffers; used to detect resizes.
    pub(crate) output_dims: (u32, u32),

    // ── Pipeline (built lazily on first render_frame) ────────────────────────
    pub(crate) scene_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) frame_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) pipeline:  Option<wgpu::ComputePipeline>,
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

        // A 1-pixel black env map dummy keeps the GPU binding slot valid
        // before any real environment is loaded.
        use wgpu::util::DeviceExt;
        let env_pixels_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("env_pixels_dummy"),
            contents: bytemuck::cast_slice(&[0.0f32; 4]),  // one vec4<f32>(0,0,0,1)
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self {
            device,
            queue,
            bvh_nodes_buf:  None,
            bvh_tris_buf:   None,
            materials_buf:  None,
            tex_data_buf:   None,
            tex_info_buf:   None,
            env_pixels_buf,
            env_dims:       (0, 0),
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

        // Pack all textures into a single flat storage buffer.
        // Each pixel becomes one u32 with R in the low byte and A in the high
        // byte, matching the order the WGSL shader unpacks them. A separate
        // metadata buffer records where each texture starts in this flat array.
        let mut tex_data: Vec<u32> = Vec::new();
        let mut tex_info: Vec<GpuTextureInfo> = Vec::new();

        for texture in textures {
            let offset = tex_data.len() as u32;
            // Pack RGBA8 bytes → u32: [R, G, B, A] → R | G<<8 | B<<16 | A<<24.
            // Using an explicit loop avoids bytemuck alignment requirements on
            // the source Vec<u8> (whose backing store may only be 1-byte aligned).
            tex_data.extend(texture.data.chunks_exact(4).map(|c| {
                (c[0] as u32) | ((c[1] as u32) << 8) | ((c[2] as u32) << 16) | ((c[3] as u32) << 24)
            }));
            tex_info.push(GpuTextureInfo {
                offset, width: texture.width, height: texture.height, _pad: 0,
            });
        }

        // WebGPU requires non-zero buffer sizes even when nothing uses them.
        if tex_data.is_empty() { tex_data.push(0); }
        if tex_info.is_empty() { tex_info.push(GpuTextureInfo { offset: 0, width: 0, height: 0, _pad: 0 }); }

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
    /// After this call the next `render_frame` will use the env map for the
    /// background and ambient lighting. Can be called multiple times to swap
    /// environments; the old buffer is dropped automatically.
    pub fn upload_environment(&mut self, width: u32, height: u32, pixels: Vec<f32>) {
        use wgpu::util::DeviceExt;
        self.env_pixels_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("env_pixels"),
            contents: bytemuck::cast_slice(&pixels),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.env_dims = (width, height);
        log::info!("env map uploaded: {}×{} ({} MB)", width, height,
            pixels.len() * 4 / 1_000_000);
    }

    /// Returns `true` once [`upload_scene`] has been called.
    pub fn is_scene_loaded(&self) -> bool {
        self.bvh_nodes_buf.is_some()
    }
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
        // Group 0: scene data — six read-only storage buffers.
        let scene_bgl = self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("scene_bgl"),
                entries: &[
                    storage_ro(0), // bvh_nodes
                    storage_ro(1), // bvh_tris
                    storage_ro(2), // materials
                    storage_ro(3), // tex_data     (flat RGBA8 pixel data)
                    storage_ro(4), // tex_info     (per-texture offset/size)
                    storage_ro(5), // env_pixels   (HDR env map as vec4<f32>)
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
        let flags = if preview { 1u32 } else { 0u32 };
        let frame_data = FrameUniforms {
            width, height, sample_index, flags,
            env_width:  self.env_dims.0,
            env_height: self.env_dims.1,
            _pad0: 0, _pad1: 0,
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
            layout:  self.scene_bgl.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bvh_nodes_buf.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bvh_tris_buf.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.materials_buf.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.tex_data_buf.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.tex_info_buf.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.env_pixels_buf.as_entire_binding() },
            ],
        });

        let frame_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("frame_bg"),
            layout:  self.frame_bgl.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.camera_buf.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.frame_buf.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.output_buf.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.accum_buf.as_ref().unwrap().as_entire_binding() },
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
            pass.set_pipeline(self.pipeline.as_ref().unwrap());
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
