//! Rust rendering engine, compiled to WebAssembly.
//!
//! This crate is the core rendering engine. It is compiled to a `.wasm` binary
//! and loaded by the TypeScript frontend running in the browser. The `#[wasm_bindgen]`
//! attribute marks functions that are exported across the WASM boundary and
//! callable from JavaScript/TypeScript.
//!
//! The high-level flow is:
//! 1. The frontend calls `load_scene` to parse a GLTF file and upload the scene
//!    to the GPU.
//! 2. The frontend calls `update_camera` continuously as the user moves around,
//!    triggering low-sample preview renders.
//! 3. When the user wants a final image, the frontend calls `render` to
//!    accumulate a high-quality result.

use wasm_bindgen::prelude::*;

pub mod scene;

#[wasm_bindgen]
pub fn init_renderer() {
    // TODO: initialize the WebGPU device and rendering pipeline
}
