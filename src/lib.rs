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
//! 4. `render()` — dispatch the compute shader (Phase 3).

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
/// A single `Renderer` (GPU device + buffers) and a `Camera` (current viewpoint)
/// live here. They are created by `init_renderer` and mutated by the other API
/// functions. Thread-local because WASM runs in a single JS thread; `RefCell`
/// gives safe interior mutability without needing a `Mutex`.
#[cfg(target_arch = "wasm32")]
struct AppState {
    renderer: renderer::Renderer,
    camera:   camera::Camera,
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

    STATE.with(|s| *s.borrow_mut() = Some(AppState { renderer, camera }));

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

        state.renderer.upload_scene(&bvh, &scene.materials);
        log::info!("scene loaded");
        Ok(())
    })
}

/// Sets the camera position and orientation and uploads the new uniform to
/// the GPU.
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

            let uniform = state.camera.to_uniform();
            state.renderer.upload_camera(&uniform);
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
pub fn render(width: u32, height: u32, sample_index: u32) {
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
            state.renderer.render_frame(width, height, sample_index);
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
        // We need a reference to the renderer outside the RefCell borrow.
        // SAFETY: We take the renderer out of the Option temporarily via a
        // raw pointer — but we only hold it for the synchronous setup portion
        // (creating the staging buffer and submitting the copy command).
        // The async await happens after the borrow is released.
        s.borrow()
            .as_ref()
            .map(|state| {
                // Capture the raw pointer; we'll re-borrow through it below.
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
