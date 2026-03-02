//! Rust path-tracing engine.
//!
//! The core renderer (`accel`, `camera`, `renderer`, `scene`) is
//! frontend-agnostic: it depends only on `wgpu` (which is itself
//! cross-platform) and pure Rust. The same `Renderer` type works in a
//! browser via WASM, a native desktop app, or a headless CI binary.
//!
//! The WASM-specific surface is confined to this file. Every `#[wasm_bindgen]`
//! attribute is guarded by `#[cfg(target_arch = "wasm32")]` so that
//! native builds compile without the `wasm-bindgen` dependency at all.
//!
//! ## High-level flow
//!
//! 1. `init_renderer()` — set up logging, select a GPU adapter, open a device.
//! 2. `load_scene(bytes)` — parse GLTF, build the BVH, upload to both renderers.
//! 3. `load_environment(bytes)` — decode an HDRI `.hdr` file and upload to the GPU.
//! 4. `update_camera(...)` — update the camera uniform and upload it.
//! 5. `render()` — dispatch the path-trace compute shader (high quality).
//! 6. `raster_frame()` — draw the rasterising preview (fast, interactive).
//! 7. Per-control setters (`set_sun_azimuth`, `set_exposure`, `set_ibl_scale`, …)
//!    — update lighting parameters; both renderers pick them up on the next frame.

// On wasm32 the prelude brings #[wasm_bindgen] and JsValue into scope.
// On native this line is compiled away entirely.
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub mod accel;
pub mod camera;
pub mod renderer;
pub mod scene;

// ============================================================================
// Global app state (WASM only)
// ============================================================================

/// Everything the WASM module keeps alive for the lifetime of the page.
///
/// A single `Renderer` (GPU device + path-tracing buffers), a `RasterRenderer`
/// (GPU raster preview pipeline), and a `Camera` (current viewpoint) live here.
/// They are created by `init_renderer` and mutated by the other API functions.
/// Thread-local because WASM runs in a single JS thread; `RefCell` gives safe
/// interior mutability without needing a `Mutex`.
#[cfg(target_arch = "wasm32")]
struct AppState {
    renderer: renderer::Renderer,
    raster:   renderer::RasterRenderer,
    camera:   camera::Camera,
    /// World-space AABB of the currently loaded scene (min corner, max corner).
    /// Set by `load_scene`; used by `get_scene_bounds` so the frontend can
    /// auto-position the camera without needing a separate bounding-box pass.
    scene_bounds: Option<([f32; 3], [f32; 3])>,
    /// Quick stats captured at scene-load time for the UI info overlay.
    /// Layout: `[triangle_count, vertex_count, mesh_count, texture_count, file_size_kb]`.
    /// `None` until the first `load_scene` call; replaced on every subsequent load.
    scene_stats: Option<[u32; 5]>,
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static STATE: std::cell::RefCell<Option<AppState>> =
        const { std::cell::RefCell::new(None) };
}

// ============================================================================
// WASM API
// ============================================================================

/// Initialises the renderer: sets up logging, selects a GPU adapter, and opens
/// a WebGPU device.
///
/// Must be called once before any other function. Returns a `Promise` that
/// resolves when the GPU device is ready, or rejects with an error string if
/// WebGPU is not supported or device creation fails.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn init_renderer() -> Result<(), JsValue> {
    // Redirect Rust panics to the browser console so you see something useful
    // instead of "unreachable executed" in a minified stack trace.
    console_error_panic_hook::set_once();

    // Route log:: macros to console.log / console.error.
    console_log::init_with_level(log::Level::Info)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let renderer = renderer::Renderer::new()
        .await
        .map_err(|e| JsValue::from_str(&e))?;

    // Default camera: 5 m back, 1 m up, looking along −Z, 60° vfov, 16:9.
    // The frontend overrides this immediately via update_camera().
    let camera = camera::Camera::new(
        [0.0, 1.0, 5.0],
        std::f32::consts::PI / 3.0,
        16.0 / 9.0,
    );

    // The raster renderer shares the same GPU device/queue as the path tracer.
    // It holds its own pipeline and buffers but borrows device/queue at each call.
    let raster = renderer::RasterRenderer::new();

    STATE.with(|s| *s.borrow_mut() = Some(AppState {
        renderer,
        raster,
        camera,
        scene_bounds: None,
        scene_stats:  None,
    }));

    log::info!("renderer ready");
    Ok(())
}

/// Parses a GLTF/GLB byte array, builds the BVH, and uploads all scene data
/// to the GPU storage buffers.
///
/// Must be called after `init_renderer`. Can be called again to swap scenes;
/// the old GPU buffers are dropped and replaced.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn load_scene(bytes: &[u8]) -> Result<(), JsValue> {
    use crate::accel::bvh::Bvh;
    use crate::scene::Scene;

    let scene = Scene::from_gltf(bytes)
        .map_err(|e| JsValue::from_str(&format!("{e:?}")))?;

    let bvh = Bvh::build(&scene);

    // Capture file size here, before `bytes` is no longer in scope relative
    // to the closure below (the closure moves `scene` and `bvh`).
    let file_kb = (bytes.len() / 1024) as u32;

    STATE.with(|s| {
        let mut guard = s.borrow_mut();
        let state = guard
            .as_mut()
            .ok_or_else(|| JsValue::from_str("call init_renderer() first"))?;

        // Capture the world-space AABB from the BVH root before upload consumes it.
        // The root node (index 0) always covers the entire scene.
        let bounds = bvh.nodes.first()
            .map(|root| (root.aabb_min, root.aabb_max));

        // Upload to path-tracer (BVH + materials + textures).
        state.renderer.upload_scene(&bvh, &scene.materials, &scene.textures);

        // Upload to rasteriser (vertex/index geometry + instance transforms).
        // The rasteriser borrows the device from the path-tracer renderer —
        // both pipelines share the same underlying GPU device.
        let (device, _queue) = state.renderer.device_queue();
        state.raster.upload_scene(device, &scene);

        // Snapshot scene statistics for the UI info overlay.  We read from
        // `scene` after both upload calls (which only borrow it) so all fields
        // are still accessible.  `file_kb` was captured from `bytes.len()`
        // before entering this closure.
        let tri_count  = (scene.indices.len()  / 3) as u32;
        let vert_count = scene.vertices.len()       as u32;
        let mesh_count = scene.meshes.len()         as u32;
        let tex_count  = scene.textures.len()       as u32;
        state.scene_stats = Some([tri_count, vert_count, mesh_count, tex_count, file_kb]);

        state.scene_bounds = bounds;
        log::info!("scene loaded");
        Ok(())
    })
}

/// Loads an HDR environment map from raw `.hdr` (Radiance RGBE) bytes.
///
/// The environment map is used by both the path tracer and the rasteriser
/// preview for background sky colour, ambient lighting, and reflections in
/// metallic surfaces. Call this after `init_renderer`. Can be called again
/// to swap environments without reloading the scene.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn load_environment(bytes: &[u8]) -> Result<(), JsValue> {
    use crate::scene::environment::decode_hdr;

    let img = decode_hdr(bytes)
        .map_err(|e| JsValue::from_str(&e))?;

    STATE.with(|s| {
        let mut guard = s.borrow_mut();
        let state = guard
            .as_mut()
            .ok_or_else(|| JsValue::from_str("call init_renderer() first"))?;

        state.renderer.upload_environment(img.width, img.height, img.pixels);
        log::info!("environment loaded: {}×{}", img.width, img.height);
        Ok(())
    })
}

/// Controls whether the loaded environment map (or procedural sky) is rendered
/// as the visible background.
///
/// When `visible` is `true` (the default), the sky/env map is drawn behind all
/// geometry. In path-traced output, rays that escape the scene contribute the
/// full env/sky radiance. In the rasteriser preview, a fullscreen sky pass
/// renders the env map or procedural gradient before geometry is drawn.
///
/// When `false`, the background shows a neutral dark grey instead — a "studio
/// lighting" look where the env map still illuminates the scene (via IBL) but
/// is not visible as the backdrop. Both renderers respect this flag.
///
/// Path-tracer changes take effect on the next `render()` call; start a fresh
/// render (`sample_index = 0`) to avoid blending old and new samples.
/// Rasteriser changes take effect on the next `raster_frame()` call.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_env_background(visible: bool) {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.renderer.env_background = visible;
            log::info!("env background {}", if visible { "on" } else { "off" });
        }
    });
}

/// Removes the currently loaded HDR environment map and reverts to the
/// procedural sky gradient.
///
/// After this call both the path tracer and the rasteriser preview behave as if
/// `load_environment` had never been called: background and IBL both come from
/// the procedural sky.
///
/// This is a no-op if no environment map is currently loaded.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn unload_environment() {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.renderer.unload_environment();
        }
    });
}

/// Enables or disables Image-Based Lighting.
///
/// When `enabled` is `true` (the default), the loaded HDR environment map is
/// used for ambient lighting and reflections in both the rasteriser preview and
/// the path tracer. When `false`, both renderers fall back to the procedural sky
/// gradient even if an env map is loaded — useful for A/B comparisons or artistic
/// control.
///
/// Changes to the rasteriser preview take effect immediately on the next
/// `raster_frame()` call. Changes to the path tracer take effect on the next
/// `render()` call; start a fresh render (`sample_index = 0`) to avoid blending
/// old (IBL) and new (procedural sky) samples.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_ibl_enabled(enabled: bool) {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.renderer.ibl_enabled = enabled;
            log::info!("IBL {}", if enabled { "enabled" } else { "disabled" });
        }
    });
}

/// Sets the maximum number of path-tracing bounces per sample.
///
/// The value is clamped to `[2, 16]`: 2 is the minimum for any meaningful
/// indirect lighting (direct hit + one bounce), and 16 is enough to resolve
/// glass caustics and multi-room interiors without wasting GPU time on paths
/// that contribute almost nothing (Russian roulette handles the tail).
///
/// Changes take effect on the next `render()` call. Start a new render
/// (`sample_index = 0`) to avoid blending old (high-bounce) and new
/// (low-bounce) samples together.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_max_bounces(n: u32) {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            let clamped = n.clamp(2, 16);
            state.renderer.max_bounces = clamped;
            log::info!("max bounces → {clamped}");
        }
    });
}

/// Sets the sun azimuth angle.
///
/// `degrees` is measured in the horizontal plane from the +Z axis toward
/// the +X axis (0° = due north / +Z, 90° = due east / +X). The shader
/// converts to a world-space direction at render time, so changing this
/// does not require re-uploading any buffers.
///
/// Valid range: 0–360°. Values outside this range are accepted and wrap
/// correctly via trigonometry.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_sun_azimuth(degrees: f32) {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.renderer.sun_azimuth = degrees.to_radians();
            log::debug!("sun azimuth → {degrees}°");
        }
    });
}

/// Sets the sun elevation angle above the horizon.
///
/// `degrees` = 0 puts the sun on the horizon; `degrees` = 90 puts it
/// directly overhead. Values are clamped by the shader's trig (a negative
/// elevation would just make the sun a below-horizon fill light, which is
/// unusual but physically plausible).
///
/// Valid range: 0–90°.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_sun_elevation(degrees: f32) {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.renderer.sun_elevation = degrees.to_radians();
            log::debug!("sun elevation → {degrees}°");
        }
    });
}

/// Sets the sun intensity multiplier.
///
/// `scale` directly multiplies `SUN_RADIANCE` in the shader's NEE
/// (Next Event Estimation) sun contribution. 0.0 effectively disables the
/// analytical sun; 1.0 is the physical default; values above 1.0 simulate
/// a brighter or harsher light source.
///
/// Has no effect when an HDR environment map is active (the env map's own
/// sun is used instead via importance sampling).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_sun_intensity(scale: f32) {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.renderer.sun_intensity = scale.max(0.0);
            log::debug!("sun intensity → {scale:.2}×");
        }
    });
}

/// Sets the IBL / sky brightness multiplier.
///
/// `scale` multiplies all environment-map and procedural-sky contributions:
/// the visible background (when `env_background` is enabled), BRDF-sampled
/// escape rays, and the explicit env NEE shadow rays. It does **not** affect
/// emissive surfaces or the analytical sun.
///
/// 0.0 = pitch-black environment; 1.0 = physical default; 2.0 = twice as
/// bright. Useful for quickly brightening a dim HDR map without editing
/// the file, or dimming the sky to match an indoor lighting scenario.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_ibl_scale(scale: f32) {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.renderer.ibl_scale = scale.max(0.0);
            log::debug!("IBL scale → {scale:.2}×");
        }
    });
}

/// Sets the display exposure in EV (Exposure Value) stops.
///
/// Applied as `linear_colour × 2^stops` just before ACES tone-mapping.
/// Because this is a post-accumulation transform, changing exposure does
/// **not** invalidate the accumulated samples — the running total stays
/// intact and the next `render` → `get_pixels` call will show the new
/// brightness. This makes the exposure slider the only control that can
/// update the live image mid-render without restarting.
///
/// +1 EV = twice as bright; −1 EV = half as bright. Typical range: −3 to +3.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_exposure(stops: f32) {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.renderer.exposure = stops;
            log::debug!("exposure → {stops:+.1} EV");
        }
    });
}

/// Sets the thin-lens aperture radius in world units.
///
/// `0.0` (the default) gives a pinhole camera where everything is in focus.
/// Larger values widen the circle of confusion for geometry that is not on
/// the focal plane, producing cinematic bokeh. Changes take effect on the
/// next `render()` call with `sample_index = 0` (a fresh accumulation).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_aperture(radius: f32) {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.renderer.aperture = radius.max(0.0);
            log::debug!("aperture → {radius:.4}");
        }
    });
}

/// Sets the focal plane distance in world units.
///
/// Geometry exactly this far from the camera appears sharp; everything closer
/// or farther blurs according to the aperture setting. Only meaningful when
/// `set_aperture` has been called with a value greater than zero. Changes take
/// effect on the next `render()` call with `sample_index = 0`.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_focus_distance(dist: f32) {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.renderer.focus_dist = dist.max(0.01);
            log::debug!("focus_dist → {dist:.3}");
        }
    });
}

/// Sets the chromatic aberration strength for the path-tracer output.
///
/// Radially shifts the red channel outward and blue channel inward from the
/// image centre, mimicking the colour fringing of cheap lenses. Strength 0.0
/// disables the effect entirely (no overhead). Good visible range: 0.0–3.0.
/// Changes take effect on the next `render()` dispatch without restarting
/// accumulation — the effect is applied to the already-accumulated average.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_ca_strength(strength: f32) {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.renderer.ca_strength = strength.max(0.0);
            log::debug!("ca_strength → {strength:.3}");
        }
    });
}

/// Sets the vignette strength for the path-tracer output.
///
/// Darkens the corners of the frame in linear light (before tone-mapping).
/// 0.0 = no darkening; 1.0 = corners nearly black. Changes take effect
/// on the next dispatch without restarting accumulation.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_vignette(strength: f32) {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.renderer.vignette = strength.clamp(0.0, 2.0);
            log::debug!("vignette → {strength:.3}");
        }
    });
}

/// Sets the film grain intensity for the path-tracer output.
///
/// Adds perceptually-uniform white noise after gamma encoding. Seed is mixed
/// with the sample index so grain never accumulates into permanent artifacts.
/// 0.0 = no grain; 1.0 ≈ heavy grain. Changes take effect on the next
/// dispatch without restarting accumulation.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_grain(strength: f32) {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.renderer.grain = strength.clamp(0.0, 1.0);
            log::debug!("grain → {strength:.3}");
        }
    });
}

/// Returns the world-space axis-aligned bounding box of the loaded scene as a
/// `Float32Array` of six values: `[min_x, min_y, min_z, max_x, max_y, max_z]`.
///
/// Returns an empty array if no scene has been loaded yet. The frontend uses
/// this to auto-position the camera so the model fills the viewport sensibly.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn get_scene_bounds() -> js_sys::Float32Array {
    STATE.with(|s| {
        s.borrow()
            .as_ref()
            .and_then(|state| state.scene_bounds)
            .map(|(mn, mx)| {
                let arr = js_sys::Float32Array::new_with_length(6);
                arr.set_index(0, mn[0]); arr.set_index(1, mn[1]); arr.set_index(2, mn[2]);
                arr.set_index(3, mx[0]); arr.set_index(4, mx[1]); arr.set_index(5, mx[2]);
                arr
            })
            .unwrap_or_else(|| js_sys::Float32Array::new_with_length(0))
    })
}

/// Returns a snapshot of the most recently loaded scene's statistics as a
/// `Uint32Array` of five values:
/// `[triangle_count, vertex_count, mesh_count, texture_count, file_size_kb]`.
///
/// The file size is in whole kilobytes; divide by 1024 in JS to get MB.
/// Returns all-zeros if no scene has been loaded yet — the frontend shows
/// `—` for each field until a real scene arrives.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn get_scene_stats() -> js_sys::Uint32Array {
    STATE.with(|s| {
        let guard = s.borrow();
        let stats = guard
            .as_ref()
            .and_then(|st| st.scene_stats)
            .unwrap_or([0; 5]);
        js_sys::Uint32Array::from(stats.as_slice())
    })
}

/// Sets the camera position and orientation and uploads the new uniform to
/// the GPU (both path-tracer and raster pipelines).
///
/// - `px / py / pz` — world-space position
/// - `yaw`   — horizontal rotation in radians (0 = looking along −Z)
/// - `pitch` — vertical tilt in radians (positive = looking up)
/// - `vfov`  — vertical field of view in radians (e.g. π/3 for 60°)
/// - `aspect` — viewport width / height
///
/// Call this every frame while the camera is moving (preview mode), or once
/// before kicking off a final render.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn update_camera(
    px: f32, py: f32, pz: f32,
    yaw: f32, pitch: f32,
    vfov: f32, aspect: f32,
) {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.camera.position = [px, py, pz];
            state.camera.yaw      = yaw;
            state.camera.pitch    = pitch;
            state.camera.vfov     = vfov;
            state.camera.aspect   = aspect;

            // Path-tracer camera (ray-generation basis vectors).
            let pt_uniform = state.camera.to_uniform();
            state.renderer.upload_camera(&pt_uniform);

            // Raster camera (view + projection matrices).
            let rs_uniform = state.camera.to_raster_uniform();
            let (device, queue) = state.renderer.device_queue();
            state.raster.upload_camera(device, queue, &rs_uniform);
        }
    });
}

/// Dispatches the path-tracing compute shader for one sample pass.
///
/// `sample_index` identifies which sample this is in a progressive render:
/// - Pass `0` to start a new render (resets the accumulator).
/// - Pass `1, 2, 3, …` to accumulate additional samples into the same image.
///
/// The output buffer holds the running average after each dispatch, so calling
/// `get_pixels` after any sample gives a valid (progressively improving) image.
///
/// Requires both `init_renderer` and `load_scene` to have been called first.
/// Calling before the scene is loaded is a no-op with a console warning.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn render(width: u32, height: u32, sample_index: u32, preview: bool) {
    let ready = STATE.with(|s| {
        s.borrow()
            .as_ref()
            .is_some_and(|s| s.renderer.is_scene_loaded())
    });

    if !ready {
        log::warn!("render() called before scene is loaded — ignoring");
        return;
    }

    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.renderer.render_frame(width, height, sample_index, preview);
        }
    });
}

/// Reads the last rendered frame back from the GPU and returns it as a
/// `Uint8Array` of raw RGBA8 bytes, suitable for passing to `ImageData`
/// and drawing to a `<canvas>` element.
///
/// Must be called after `render()`. Returns an empty array if no frame has
/// been rendered yet.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn get_pixels(width: u32, height: u32) -> js_sys::Uint8Array {
    let pixels = STATE.with(|s| {
        s.borrow()
            .as_ref()
            .map(|state| {
                &state.renderer as *const renderer::Renderer
            })
    });

    let Some(renderer_ptr) = pixels else {
        return js_sys::Uint8Array::new_with_length(0);
    };

    // SAFETY: `renderer_ptr` points into `AppState::renderer`, which lives
    // inside the `STATE` thread_local for the lifetime of the page. Two
    // invariants must hold for this to be sound:
    //   1. No mutation — `get_pixels` takes `&self`, so the object is not
    //      modified while the pointer is live.
    //   2. No re-initialisation — if `init_renderer()` were called again
    //      while this future is pending it would drop the old `AppState`
    //      and dangle this pointer.  The JS API must never call
    //      `init_renderer` more than once; it is called exactly once at
    //      page load before any render futures are created.
    //
    // Both invariants are guaranteed by the single-threaded WASM execution
    // model and the JS-side calling convention, not by the type system.
    let renderer = unsafe { &*renderer_ptr };
    let bytes = renderer.get_pixels(width, height).await;
    js_sys::Uint8Array::from(bytes.as_slice())
}

/// Draws the scene using the fast rasterising preview pipeline.
///
/// This runs at 60 fps during interactive dragging. Unlike the path tracer
/// (`render`), it produces a single fully-formed frame with no progressive
/// accumulation — call `raster_get_pixels` immediately after to retrieve it.
///
/// All user-adjustable controls (sun azimuth/elevation, sun intensity, exposure,
/// IBL scale, IBL enabled/disabled, and any loaded HDR env map) are reflected in
/// the preview frame just as they are in the path-traced output.
///
/// Requires both `init_renderer` and `load_scene` to have been called first.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn raster_frame(width: u32, height: u32) {
    let ready = STATE.with(|s| {
        s.borrow()
            .as_ref()
            .is_some_and(|s| s.raster.is_scene_loaded())
    });

    if !ready {
        log::warn!("raster_frame() called before scene is loaded — ignoring");
        return;
    }

    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            // ── Build RasterFrameUniforms from current renderer state ─────────
            // Convert azimuth + elevation to a world-space unit direction vector.
            // Convention: azimuth measured from +Z toward +X (same as the
            // path-tracer's frame.sun_azimuth / frame.sun_elevation fields).
            let az     = state.renderer.sun_azimuth;
            let el     = state.renderer.sun_elevation;
            let cos_el = el.cos();
            let sun_dir = [az.sin() * cos_el, el.sin(), az.cos() * cos_el];

            // env_available = 1 only when an HDR map is loaded AND IBL is on.
            // env_dims is (0,0) when using the dummy 1×1 black pixel, so
            // checking for non-zero is sufficient to detect "real env loaded".
            let env_dims  = state.renderer.env_dims();
            let env_avail = u32::from(env_dims != (0, 0) && state.renderer.ibl_enabled);

            let frame = renderer::raster::RasterFrameUniforms {
                sun_dir,
                sun_intensity:  state.renderer.sun_intensity,
                exposure:       state.renderer.exposure,
                ibl_scale:      state.renderer.ibl_scale,
                env_available:  env_avail,
                env_width:      env_dims.0,
                env_height:     env_dims.1,
                // env_background mirrors the path tracer flag: when true, the sky
                // pass samples the env map / procedural sky as the visible backdrop.
                env_background: u32::from(state.renderer.env_background),
                _pad:           [0; 2],
            };

            // render_frame borrows device+queue from the path-tracer renderer
            // (both pipelines share the same underlying GPU device), and borrows
            // env_pixels so the env map is never duplicated in GPU memory.
            let env_pixels = state.renderer.env_pixels_buf();
            let (device, queue) = state.renderer.device_queue();
            state.raster.render_frame(device, queue, width, height, &frame, env_pixels);
        }
    });
}

/// Reads the last rasterised frame back from the GPU and returns it as a
/// `Uint8Array` of raw RGBA8 bytes.
///
/// Must be called immediately after `raster_frame()`.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn raster_get_pixels(width: u32, height: u32) -> js_sys::Uint8Array {
    // Capture raw pointers before the async boundary so we can hold references
    // across the `.await` without fighting the borrow checker.
    let ptrs = STATE.with(|s| {
        s.borrow().as_ref().map(|state| (
            &state.raster           as *const renderer::RasterRenderer,
            &state.renderer.device  as *const wgpu::Device,
            &state.renderer.queue   as *const wgpu::Queue,
        ))
    });

    let Some((raster_ptr, device_ptr, queue_ptr)) = ptrs else {
        return js_sys::Uint8Array::new_with_length(0);
    };

    // SAFETY: All three pointers reference objects inside `AppState`, which
    // lives in the `STATE` thread_local for the lifetime of the page.
    // Three invariants must hold:
    //   1. No mutation — all three methods are `&self`; the objects are not
    //      modified while these raw references are live.
    //   2. No re-initialisation — `init_renderer()` must not be called again
    //      while this future is pending; doing so would drop `AppState` and
    //      dangle all three pointers.
    //   3. Single-threaded — WASM runs on a single JS thread, so there is
    //      no concurrent access even without a lock.
    // The JS calling convention guarantees invariants 2 and 3.
    let raster = unsafe { &*raster_ptr };
    let device = unsafe { &*device_ptr };
    let queue  = unsafe { &*queue_ptr };

    let bytes = raster.get_pixels(device, queue, width, height).await;
    js_sys::Uint8Array::from(bytes.as_slice())
}
