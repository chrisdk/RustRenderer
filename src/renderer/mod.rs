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

// ============================================================================
// Shader tests
// ============================================================================
//
// These tests validate WGSL shaders using naga — the same shader compiler that
// wgpu (and the browser's WebGPU implementation) uses internally.
//
// Why? Two categories of bug are invisible to `cargo build` and only surface at
// runtime in the browser, typically as a silent black canvas:
//
//   1. **Invalid WGSL** — e.g. using `array<u32, 3>` in `var<uniform>`.
//      WGSL's uniform address space requires array element types to have
//      16-byte alignment; `u32` has only 4-byte alignment. The Rust code
//      compiles fine, the shader module is created without a Rust-side
//      error (WebGPU reports errors asynchronously), and every draw call
//      silently becomes a no-op.
//
//   2. **Rust/WGSL layout mismatch** — e.g. adding a field to a Rust struct
//      and forgetting to add it to the matching WGSL struct. The Rust layout
//      test passes (the struct has the right offsets), the WGSL compiles (the
//      shader has its own self-consistent layout), but the bytes the GPU reads
//      from the uniform buffer don't match what Rust wrote. Result: wrong sun
//      direction, wrong exposure, garbled material data — all silent.
//
// These tests run on the CPU in the normal `cargo test` pass, with no GPU or
// browser required. They catch both classes of bug before a build is needed.

#[cfg(test)]
mod shader_tests {
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Parse a WGSL source string and fully validate it with naga.
    ///
    /// Panics with the naga error message on parse or validation failure,
    /// making the test output point directly at the problem.
    fn validate(label: &str, source: &str) -> naga::Module {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{label}: WGSL parse error:\n{e:#?}"));
        let mut v = Validator::new(ValidationFlags::all(), Capabilities::default());
        v.validate(&module)
            .unwrap_or_else(|e| panic!("{label}: WGSL validation error:\n{e:#?}"));
        module
    }

    /// Find a named struct's member list inside a naga module.
    ///
    /// Panics if the struct is absent — a missing struct is itself a bug worth
    /// surfacing loudly.
    fn struct_members<'m>(module: &'m naga::Module, name: &str) -> &'m [naga::StructMember] {
        for (_, ty) in module.types.iter() {
            if ty.name.as_deref() == Some(name) {
                if let naga::TypeInner::Struct { members, .. } = &ty.inner {
                    return members;
                }
            }
        }
        panic!("struct {name:?} not found in WGSL module");
    }

    /// Look up one member's byte offset by name.
    ///
    /// Panics if the field is missing — if a field was removed from WGSL but
    /// not from Rust (or vice-versa) we want to know about it immediately.
    fn field_offset(members: &[naga::StructMember], field: &str) -> u32 {
        members
            .iter()
            .find(|m| m.name.as_deref() == Some(field))
            .unwrap_or_else(|| panic!("field {field:?} not found in WGSL struct"))
            .offset
    }

    // ── Shader validity ───────────────────────────────────────────────────────

    /// `raster.wgsl` must parse and validate cleanly with naga.
    ///
    /// This catches any WGSL type error that would cause silent failure in the
    /// browser. The most dangerous class: types that are valid in
    /// `var<storage>` but rejected in `var<uniform>` (e.g. `array<u32, N>`,
    /// which requires 16-byte element alignment in the uniform address space).
    #[test]
    fn test_raster_wgsl_validates() {
        validate("raster.wgsl", include_str!("../shaders/raster.wgsl"));
    }

    /// `trace.wgsl` must parse and validate cleanly with naga.
    #[test]
    fn test_trace_wgsl_validates() {
        validate("trace.wgsl", include_str!("../shaders/trace.wgsl"));
    }

    /// Every integer-typed field in a vertex shader's output struct must carry
    /// an explicit `@interpolate(flat)` attribute in the WGSL **source text**.
    ///
    /// ## Why a source-text check, not an IR check?
    ///
    /// Naga infers `Interpolation::Flat` for integer-typed location bindings
    /// automatically, even when the attribute is absent from the source. So an
    /// IR-level check (inspecting `naga::Binding::Location { interpolation }`)
    /// always sees `Some(Flat)` and would pass even on the broken shader.
    ///
    /// Chrome's Tint compiler (the WebGPU reference implementation) does *not*
    /// do this inference — it requires the explicit annotation and rejects the
    /// shader at parse time with:
    ///
    /// > "integral user-defined vertex outputs must have a '@interpolate(flat)' attribute"
    ///
    /// The mismatch means a shader passes `cargo test` (Naga) and then
    /// silently produces a blank canvas on Chrome. Checking the source text
    /// directly closes that gap without needing Tint installed.
    ///
    /// The bug that prompted this test: `mat_index: u32` in `VertexOutput` was
    /// missing `@interpolate(flat)`, discovered only when testing on Chrome M1.
    #[test]
    fn test_raster_vertex_integer_outputs_are_flat() {
        let src = include_str!("../shaders/raster.wgsl");

        // Locate the VertexOutput struct body. The output of the vertex
        // entry point is the only struct whose fields need @interpolate(flat).
        let struct_start = src.find("struct VertexOutput {")
            .expect("VertexOutput struct not found in raster.wgsl");
        let brace_open  = struct_start + src[struct_start..].find('{').unwrap() + 1;
        let brace_close = brace_open   + src[brace_open..].find('}').unwrap();
        let struct_body = &src[brace_open..brace_close];

        // Integer primitive types that, per WGSL spec §11.3.1, cannot be
        // interpolated across a triangle and require @interpolate(flat).
        let integer_types = [
            "u32", "i32",
            "vec2<u32>", "vec3<u32>", "vec4<u32>",
            "vec2<i32>", "vec3<i32>", "vec4<i32>",
        ];

        for line in struct_body.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("//") { continue; }

            // A field line looks like:
            //   @location(N) @interpolate(flat) name: u32,
            // or it might have a trailing comment. We check:
            //   - has @location (so it's an inter-stage varying, not a builtin)
            //   - has an integer type
            //   - has @interpolate(flat)
            let has_location = line.contains("@location(");
            let is_integer   = integer_types.iter().any(|t| {
                // Match `: TYPE` with optional trailing comma/comment.
                line.contains(&format!(": {t},")) || line.contains(&format!(": {t} "))
                    || line.ends_with(&format!(": {t}"))
            });

            if has_location && is_integer {
                assert!(
                    line.contains("@interpolate(flat)"),
                    "VertexOutput field is missing @interpolate(flat):\n  {line}\n\
                     Chrome/Tint requires an explicit @interpolate(flat) on integer \
                     vertex outputs. Naga infers it silently, masking the error until \
                     the shader hits Chrome at runtime.",
                );
            }
        }
    }

    // ── Rust ↔ WGSL layout cross-checks ──────────────────────────────────────
    //
    // The Rust structs already have `offset_of!` tests in their own modules,
    // confirming the Rust side is self-consistent. These tests ask a different
    // question: does the *WGSL* struct agree with the Rust struct, field by
    // field? naga computes the authoritative WGSL offsets using the same
    // layout rules the GPU will actually apply at runtime.
    //
    // If a field is added to one side and forgotten on the other, the offset
    // check for the first mismatched field will fail and name the culprit.

    /// `RasterFrame` in `raster.wgsl` must have the same field offsets as
    /// `RasterFrameUniforms` in Rust.
    ///
    /// The bug that prompted these tests: `_pad: array<u32, 3>` was written
    /// in the WGSL struct but is illegal in `var<uniform>`. naga rejects it,
    /// so this test would have failed before the fix was applied — catching
    /// the bug at `cargo test` time instead of at runtime in the browser.
    #[test]
    fn test_raster_frame_wgsl_layout_matches_rust() {
        use crate::renderer::raster::RasterFrameUniforms;
        use std::mem::offset_of;

        let module = validate("raster.wgsl", include_str!("../shaders/raster.wgsl"));
        let m = struct_members(&module, "RasterFrame");

        assert_eq!(field_offset(m, "sun_dir"),        offset_of!(RasterFrameUniforms, sun_dir)        as u32);
        assert_eq!(field_offset(m, "sun_intensity"),  offset_of!(RasterFrameUniforms, sun_intensity)  as u32);
        assert_eq!(field_offset(m, "exposure"),       offset_of!(RasterFrameUniforms, exposure)       as u32);
        assert_eq!(field_offset(m, "ibl_scale"),      offset_of!(RasterFrameUniforms, ibl_scale)      as u32);
        assert_eq!(field_offset(m, "env_available"),  offset_of!(RasterFrameUniforms, env_available)  as u32);
        assert_eq!(field_offset(m, "env_width"),      offset_of!(RasterFrameUniforms, env_width)      as u32);
        assert_eq!(field_offset(m, "env_height"),     offset_of!(RasterFrameUniforms, env_height)     as u32);
        assert_eq!(field_offset(m, "env_background"), offset_of!(RasterFrameUniforms, env_background) as u32);
    }

    /// `FrameUniforms` in `trace.wgsl` must have the same field offsets as
    /// `FrameUniforms` in Rust (`src/renderer/gpu.rs`).
    #[test]
    fn test_frame_uniforms_wgsl_layout_matches_rust() {
        use crate::renderer::gpu::FrameUniforms;
        use std::mem::offset_of;

        let module = validate("trace.wgsl", include_str!("../shaders/trace.wgsl"));
        let m = struct_members(&module, "FrameUniforms");

        assert_eq!(field_offset(m, "width"),             offset_of!(FrameUniforms, width)             as u32);
        assert_eq!(field_offset(m, "height"),            offset_of!(FrameUniforms, height)            as u32);
        assert_eq!(field_offset(m, "sample_index"),      offset_of!(FrameUniforms, sample_index)      as u32);
        assert_eq!(field_offset(m, "flags"),             offset_of!(FrameUniforms, flags)             as u32);
        assert_eq!(field_offset(m, "env_width"),         offset_of!(FrameUniforms, env_width)         as u32);
        assert_eq!(field_offset(m, "env_height"),        offset_of!(FrameUniforms, env_height)        as u32);
        assert_eq!(field_offset(m, "env_background"),    offset_of!(FrameUniforms, env_background)    as u32);
        assert_eq!(field_offset(m, "env_sample_weight"), offset_of!(FrameUniforms, env_sample_weight) as u32);
        assert_eq!(field_offset(m, "max_bounces"),       offset_of!(FrameUniforms, max_bounces)       as u32);
        assert_eq!(field_offset(m, "sun_azimuth"),       offset_of!(FrameUniforms, sun_azimuth)       as u32);
        assert_eq!(field_offset(m, "sun_elevation"),     offset_of!(FrameUniforms, sun_elevation)     as u32);
        assert_eq!(field_offset(m, "sun_intensity"),     offset_of!(FrameUniforms, sun_intensity)     as u32);
        assert_eq!(field_offset(m, "ibl_scale"),         offset_of!(FrameUniforms, ibl_scale)         as u32);
        assert_eq!(field_offset(m, "exposure"),          offset_of!(FrameUniforms, exposure)          as u32);
    }
}
