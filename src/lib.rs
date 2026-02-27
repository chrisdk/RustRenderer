//! Rust path-tracing engine.
//!
//! The core renderer (`accel`, `camera`, `renderer`, `scene`) is
//! frontend-agnostic: it depends only on `wgpu` (which is itself
//! cross-platform) and pure Rust. The same `Renderer` type works in a
//! browser via WASM, a native desktop app, or a headless CI binary.
//!
//! The WASM-specific surface is confined to this file. Every `#[wasm_bindgen]`
//! attribute is guarded by `#[cfg_attr(target_arch = "wasm32", ...)]` so that
//! native builds compile without the `wasm-bindgen` dependency at all.
//!
//! ## High-level flow
//!
//! 1. `init_renderer()` — select a GPU adapter and open a device.
//! 2. `load_scene(bytes)` — parse GLTF, build the BVH, upload to the GPU.
//! 3. `update_camera(...)` — update the camera and kick off a preview render.
//! 4. `render()` — accumulate samples until the image converges.

// On wasm32 the prelude brings #[wasm_bindgen] and JsValue into scope.
// On native this line is compiled away entirely.
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub mod accel;
pub mod camera;
pub mod renderer;
pub mod scene;

/// Initialises the renderer: sets up logging and acquires a GPU device.
///
/// Must be called once before any other function. On WASM this is exposed
/// to JavaScript as an async function returning a Promise.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub fn init_renderer() {
    // TODO: initialize the WebGPU device and rendering pipeline
}
