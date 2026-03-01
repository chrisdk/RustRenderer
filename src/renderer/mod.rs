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
// Shared helpers
// ============================================================================

/// Packs scene textures into a pair of GPU-ready storage buffers.
///
/// Returns `(tex_data, tex_info)`:
///
/// - `tex_data` — flat pixel buffer; each RGBA8 pixel is one `u32` packed as
///   `R | G<<8 | B<<16 | A<<24`, matching the order `sample_tex()` unpacks
///   in both `trace.wgsl` and `raster.wgsl`.
/// - `tex_info` — one [`GpuTexInfo`] per texture recording its pixel offset
///   into `tex_data` plus its dimensions.
///
/// Both vectors are guaranteed non-empty because WebGPU forbids zero-size
/// storage buffers. A dummy zero-pixel / zero-dimension sentinel is appended
/// when the texture list is empty.
///
/// Used by both the path-tracer (`gpu::Renderer`) and the rasterizer
/// (`raster::RasterRenderer`) so that an identical packing format is kept in
/// a single place and tested once.
pub(crate) fn pack_textures(
    textures: &[crate::scene::texture::Texture],
) -> (Vec<u32>, Vec<GpuTexInfo>) {
    let mut tex_data: Vec<u32>        = Vec::new();
    let mut tex_info: Vec<GpuTexInfo> = Vec::new();

    for texture in textures {
        let offset = tex_data.len() as u32;
        // Pack RGBA8 bytes → u32: [R, G, B, A] → R | G<<8 | B<<16 | A<<24.
        // Explicit loop (no bytemuck cast) because the source Vec<u8> may
        // only be 1-byte aligned — cast_slice requires at least 4-byte alignment.
        tex_data.extend(texture.data.chunks_exact(4).map(|c| {
            (c[0] as u32) | ((c[1] as u32) << 8) | ((c[2] as u32) << 16) | ((c[3] as u32) << 24)
        }));
        tex_info.push(GpuTexInfo { offset, width: texture.width, height: texture.height, _pad: 0 });
    }

    // Guard against zero-size buffers — WebGPU rejects them unconditionally.
    if tex_data.is_empty() { tex_data.push(0); }
    if tex_info.is_empty() { tex_info.push(GpuTexInfo { offset: 0, width: 0, height: 0, _pad: 0 }); }

    (tex_data, tex_info)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    use std::mem::{offset_of, size_of};

    /// `pack_textures` is the single source of truth for the pixel packing used
    /// by both GPU pipelines. This test verifies the RGBA8 → u32 encoding and
    /// the zero-size guard (empty input must still produce non-empty buffers).
    #[test]
    fn test_pack_textures_encoding_and_empty_guard() {
        use crate::scene::texture::Texture;

        // ── Single 1×1 red pixel (fully opaque) ──────────────────────────────
        let red = Texture { width: 1, height: 1, data: vec![255, 0, 0, 255] };
        let (data, info) = pack_textures(&[red]);
        assert_eq!(data.len(), 1);
        // R=0xFF, G=0x00, B=0x00, A=0xFF → 0xFF00_00FF
        assert_eq!(data[0], 0xFF00_00FF, "red pixel packing");
        assert_eq!(info[0].offset, 0);
        assert_eq!(info[0].width,  1);
        assert_eq!(info[0].height, 1);

        // ── Two textures — offsets must be sequential ─────────────────────────
        let t1 = Texture { width: 1, height: 1, data: vec![10, 20, 30, 255] };
        let t2 = Texture { width: 2, height: 1, data: vec![0, 0, 0, 0,  1, 2, 3, 4] };
        let (data2, info2) = pack_textures(&[t1, t2]);
        assert_eq!(data2.len(), 3, "1 pixel + 2 pixels = 3 u32s");
        assert_eq!(info2[0].offset, 0, "first texture starts at 0");
        assert_eq!(info2[1].offset, 1, "second texture starts after first");

        // ── Empty input must produce non-empty dummy buffers ──────────────────
        let (empty_data, empty_info) = pack_textures(&[]);
        assert_eq!(empty_data.len(), 1, "zero-size guard: tex_data must have ≥1 element");
        assert_eq!(empty_info.len(), 1, "zero-size guard: tex_info must have ≥1 element");
    }

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
