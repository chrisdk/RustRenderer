//! The renderer: CPU ground-truth traversal and the building blocks for the
//! GPU compute shader.
//!
//! This module is organised in layers, from the bottom up:
//!
//! | Module      | What it does                                            |
//! |-------------|----------------------------------------------------------|
//! | `math`      | Vector math helpers (`dot`, `cross`, `reflect`, …)      |
//! | `intersect` | Ray–AABB and ray–triangle intersection tests            |
//! | `traverse`  | BVH traversal; produces a [`HitRecord`] per ray         |
//!
//! The CPU implementations here are the ground truth for the WGSL compute
//! shader. When the GPU produces a wrong pixel, reproduce the failing ray in
//! Rust first — you'll have `println!`, a debugger, and unit tests to work
//! with.

pub mod math;
pub mod intersect;
pub mod traverse;

pub use traverse::HitRecord;
pub use traverse::intersect_bvh;
