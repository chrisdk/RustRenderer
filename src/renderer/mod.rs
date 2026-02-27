//! The renderer: CPU ground-truth traversal and the GPU pipeline.
//!
//! This module is organised in layers, from the bottom up:
//!
//! | Module      | What it does                                              |
//! |-------------|-----------------------------------------------------------|
//! | `math`      | Vector math helpers (`dot`, `cross`, `reflect`, …)        |
//! | `intersect` | Ray–AABB and ray–triangle intersection tests              |
//! | `traverse`  | BVH traversal; produces a [`HitRecord`] per ray           |
//! | `gpu`       | [`Renderer`]: GPU device, buffers, and (soon) the shader  |
//!
//! **Debugging tip**: when the GPU shader produces a wrong pixel, reproduce
//! the failing ray in `traverse.rs` first — you'll have `println!`, a
//! debugger, and 68 unit tests to work with before touching WGSL.

pub mod gpu;
pub mod intersect;
pub mod math;
pub mod traverse;

pub use gpu::Renderer;
pub use traverse::HitRecord;
pub use traverse::intersect_bvh;
