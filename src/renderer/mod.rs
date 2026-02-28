//! The renderer: CPU ground-truth traversal and the GPU pipeline.
//!
//! This module is organised in layers, from the bottom up:
//!
//! | Module      | What it does                                              |
//! |-------------|-----------------------------------------------------------|
//! | `math`      | Vector math helpers (`dot`, `cross`, `reflect`, …)        |
//! | `intersect` | Ray–AABB and ray–triangle intersection tests              |
//! | `traverse`  | BVH traversal; produces a [`HitRecord`] per ray           |
//! | `gpu`       | [`Renderer`]: GPU device, buffers, path-tracer pipeline   |
//! | `raster`    | [`RasterRenderer`]: hardware rasterizer, 60 fps preview   |
//!
//! Shared GPU types used by both pipelines live here in `mod.rs` so that
//! neither `gpu` nor `raster` needs to import from its sibling.
//!
//! **Debugging tip**: when the GPU shader produces a wrong pixel, reproduce
//! the failing ray in `traverse.rs` first — you'll have `println!`, a
//! debugger, and unit tests to work with before touching WGSL.

pub mod gpu;
pub mod intersect;
pub mod math;
pub mod raster;
pub mod traverse;

pub use gpu::Renderer;
pub use raster::RasterRenderer;
pub use traverse::HitRecord;
pub use traverse::intersect_bvh;

// ============================================================================
// Shared GPU types
// ============================================================================

/// Per-texture metadata uploaded alongside the flat pixel-data buffer.
///
/// Both the path-tracer ([`gpu`]) and the rasterizer ([`raster`]) pack all
/// scene textures into a single `tex_data` storage buffer (one `u32` per
/// RGBA8 pixel, in R|G<<8|B<<16|A<<24 order). This struct sits in a parallel
/// `tex_info` buffer, one entry per texture, telling the shader where in
/// `tex_data` each texture starts and what its dimensions are.
///
/// **Must match the `TexInfo` struct in both `trace.wgsl` and `raster.wgsl`.**
/// The layout test below is the authoritative check for all three.
///
/// 16 bytes — a multiple of 16, as required for WGSL storage-buffer arrays.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuTexInfo {
    /// Pixel index (not byte offset) into `tex_data` where this texture starts.
    pub(crate) offset: u32,
    pub(crate) width:  u32,
    pub(crate) height: u32,
    pub(crate) _pad:   u32,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    use std::mem::{offset_of, size_of};

    /// `GpuTexInfo` is used by both the path-tracer and the rasterizer.
    /// It must match the `TexInfo` struct in `trace.wgsl` and `raster.wgsl`
    /// byte-for-byte. A silent layout change would corrupt texture sampling
    /// in both pipelines simultaneously.
    #[test]
    fn test_gpu_tex_info_layout() {
        assert_eq!(size_of::<GpuTexInfo>(), 16,
            "GpuTexInfo must be 16 bytes (multiple of 16 for WGSL storage arrays)");
        assert_eq!(offset_of!(GpuTexInfo, offset), 0);
        assert_eq!(offset_of!(GpuTexInfo, width),  4);
        assert_eq!(offset_of!(GpuTexInfo, height), 8);
        // _pad at byte 12; verify Pod is satisfied (no uninit bytes).
        let _: &[u8] = bytemuck::bytes_of(&GpuTexInfo::zeroed());
    }
}
