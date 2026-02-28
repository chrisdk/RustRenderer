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
//! 2. `load_scene(bytes)` — parse GLTF, build the BVH, upload to the GPU.
//! 3. `update_camera(...)` — update the camera uniform and upload it.
//! 4. `render()` — dispatch the path-trace compute shader (high quality).
//! 5. `raster_frame()` — draw the rasterising preview (fast, interactive).

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
        // Borrows device/queue from the path-tracer renderer — they share the
        // same underlying GPU device.
        state.raster.upload_scene(
            &state.renderer.device,
            &state.renderer.queue,
            &scene,
        );

        state.scene_bounds = bounds;
        log::info!("scene loaded");
        Ok(())
    })
}

/// Loads an HDR environment map from raw `.hdr` (Radiance RGBE) bytes.
///
/// The environment map is used by the path tracer for background sky colour,
/// ambient lighting, and reflections in metallic surfaces. Call this after
/// `init_renderer`. Can be called again to swap environments without reloading
/// the scene.
///
/// The rasteriser continues to use its procedural sky gradient; this only
/// affects the path-traced output from the "Render" button.
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

/// Enables or disables Image-Based Lighting for path-traced renders.
///
/// When `enabled` is `true` (the default), the loaded HDR environment map
/// is used for the background, ambient lighting, and reflections. When
/// `false`, the path tracer falls back to the procedural sky gradient even
/// if an env map is loaded — useful for A/B comparisons or artistic control.
///
/// Has no effect on the rasteriser preview, which always uses its own
/// procedural sky. Changes take effect on the next `render()` call; the
/// accumulator is automatically reset so you see a clean result.
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
            state.raster.upload_camera(
                &state.renderer.device,
                &state.renderer.queue,
                &rs_uniform,
            );
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
            .map_or(false, |s| s.renderer.is_scene_loaded())
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

    // SAFETY: The renderer lives as long as STATE, and STATE is thread_local.
    // We don't mutate it here (get_pixels takes &self), and WASM is
    // single-threaded, so no concurrent access is possible.
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
/// Requires both `init_renderer` and `load_scene` to have been called first.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn raster_frame(width: u32, height: u32) {
    let ready = STATE.with(|s| {
        s.borrow()
            .as_ref()
            .map_or(false, |s| s.raster.is_scene_loaded())
    });

    if !ready {
        log::warn!("raster_frame() called before scene is loaded — ignoring");
        return;
    }

    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            // render_frame takes device+queue as parameters; borrow them
            // from the path-tracer renderer (same GPU device).
            state.raster.render_frame(
                &state.renderer.device,
                &state.renderer.queue,
                width,
                height,
            );
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

    // SAFETY: All three objects live in STATE (thread_local). WASM is
    // single-threaded, so no concurrent mutation occurs, and none of these
    // are mutated by get_pixels (it takes &self / shared references).
    let raster = unsafe { &*raster_ptr };
    let device = unsafe { &*device_ptr };
    let queue  = unsafe { &*queue_ptr };

    let bytes = raster.get_pixels(device, queue, width, height).await;
    js_sys::Uint8Array::from(bytes.as_slice())
}
