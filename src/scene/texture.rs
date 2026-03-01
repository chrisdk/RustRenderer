use gltf::image::Format;

/// A decoded image stored in CPU memory, ready to be uploaded to the GPU.
///
/// Regardless of the source format in the GLTF file, all textures are
/// normalized to RGBA8 (four bytes per pixel: red, green, blue, alpha, each
/// 0–255). This uniform format simplifies GPU upload since every texture
/// can be handled identically, with no per-texture format negotiation.
pub struct Texture {
    pub width:  u32,
    pub height: u32,

    /// Raw pixel data in RGBA8 format, laid out row by row from top-left.
    /// Total length is always width * height * 4 bytes.
    pub data: Vec<u8>,
}

impl Texture {
    pub fn from_gltf_image(image: &gltf::image::Data) -> Self {
        Self {
            width:  image.width,
            height: image.height,
            data:   to_rgba8(&image.pixels, image.format, image.width, image.height),
        }
    }
}

/// Converts raw pixel data from any GLTF image format into RGBA8.
///
/// GLTF files can contain images in several formats depending on what the
/// authoring tool wrote. We normalize them all to RGBA8 here so the rest
/// of the renderer only ever has to deal with one format.
fn to_rgba8(pixels: &[u8], format: Format, width: u32, height: u32) -> Vec<u8> {
    let n = (width * height) as usize;
    match format {
        // Already in the format we want — just copy.
        Format::R8G8B8A8 => pixels.to_vec(),

        // RGB without alpha — append a fully-opaque alpha byte to each pixel.
        Format::R8G8B8 => {
            let mut out = Vec::with_capacity(n * 4);
            for rgb in pixels.chunks_exact(3) {
                out.extend_from_slice(rgb);
                out.push(255);
            }
            out
        }

        // Two-channel (e.g. metallic+roughness packed textures).
        // Pad with a zero blue channel and fully-opaque alpha.
        Format::R8G8 => {
            let mut out = Vec::with_capacity(n * 4);
            for rg in pixels.chunks_exact(2) {
                out.extend_from_slice(&[rg[0], rg[1], 0, 255]);
            }
            out
        }

        // Single-channel (e.g. a grayscale roughness or occlusion map).
        // Replicate the value into R, G, and B to produce a grayscale RGBA pixel.
        Format::R8 => {
            let mut out = Vec::with_capacity(n * 4);
            for &r in pixels {
                out.extend_from_slice(&[r, r, r, 255]);
            }
            out
        }

        // 16-bit formats offer higher precision than 8-bit but we downsample
        // them to 8-bit here. Values are stored little-endian (low byte first),
        // so the high byte at index 1 captures the most significant bits and
        // gives a good 8-bit approximation of the original value.
        Format::R16G16B16A16 => {
            let mut out = Vec::with_capacity(n * 4);
            for c in pixels.chunks_exact(8) {
                out.extend_from_slice(&[c[1], c[3], c[5], c[7]]);
            }
            out
        }
        Format::R16G16B16 => {
            let mut out = Vec::with_capacity(n * 4);
            for c in pixels.chunks_exact(6) {
                out.extend_from_slice(&[c[1], c[3], c[5], 255]);
            }
            out
        }
        Format::R16G16 => {
            let mut out = Vec::with_capacity(n * 4);
            for c in pixels.chunks_exact(4) {
                out.extend_from_slice(&[c[1], c[3], 0, 255]);
            }
            out
        }
        Format::R16 => {
            let mut out = Vec::with_capacity(n * 4);
            for c in pixels.chunks_exact(2) {
                out.extend_from_slice(&[c[1], c[1], c[1], 255]);
            }
            out
        }

        // High-dynamic-range floating-point formats (R32F, R32G32B32F, etc.)
        // can appear as inline material textures in some GLTF files.
        // Note: environment maps follow a completely separate decode path
        // (decode_hdr in environment.rs, called by the load_environment API)
        // and never reach this function. This arm only fires for HDR textures
        // embedded directly inside GLTF materials, which is rare in practice.
        // Return a solid white placeholder until tone-mapping is added here.
        _ => vec![255u8; n * 4],
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use gltf::image::Format;

    /// Helper: convert a single-pixel input and assert the RGBA8 output.
    fn check_one_pixel(pixels: &[u8], format: Format, expected: [u8; 4]) {
        let got = to_rgba8(pixels, format, 1, 1);
        assert_eq!(got.as_slice(), &expected,
            "{format:?}: expected {expected:?}, got {got:?}");
    }

    // ── Pass-through ───────────────────────────────────────────────────────

    /// R8G8B8A8 is already in the target format; the bytes must come through
    /// completely unchanged. Any modification here would corrupt every texture
    /// that an authoring tool stored in native RGBA8.
    #[test]
    fn test_r8g8b8a8_passthrough() {
        check_one_pixel(&[10, 20, 30, 128], Format::R8G8B8A8, [10, 20, 30, 128]);
        // Boundary values.
        check_one_pixel(&[0, 0, 0, 0],       Format::R8G8B8A8, [0, 0, 0, 0]);
        check_one_pixel(&[255, 255, 255, 255], Format::R8G8B8A8, [255, 255, 255, 255]);
    }

    // ── 8-bit source formats ───────────────────────────────────────────────

    /// R8G8B8 → RGBA8: alpha must be filled with 255 (fully opaque).
    #[test]
    fn test_r8g8b8_adds_opaque_alpha() {
        check_one_pixel(&[10, 20, 30], Format::R8G8B8, [10, 20, 30, 255]);
    }

    /// R8G8 (e.g. packed metallic+roughness) → RGBA8: blue must be 0, alpha 255.
    #[test]
    fn test_r8g8_pads_blue_and_alpha() {
        check_one_pixel(&[10, 20], Format::R8G8, [10, 20, 0, 255]);
    }

    /// R8 (grayscale) → RGBA8: the single value is replicated into R, G, and B.
    /// A pure-white occlusion map must become (255,255,255,255), not (255,0,0,255).
    #[test]
    fn test_r8_replicates_to_grayscale() {
        check_one_pixel(&[0],   Format::R8, [0,   0,   0,   255]);
        check_one_pixel(&[128], Format::R8, [128, 128, 128, 255]);
        check_one_pixel(&[255], Format::R8, [255, 255, 255, 255]);
    }

    // ── 16-bit source formats (downsampled to 8-bit) ───────────────────────
    //
    // 16-bit values are stored little-endian: the low byte comes first, the
    // high byte second. We keep only the high byte as a fast 8-bit approximation.

    /// R16G16B16A16 → RGBA8: take the high byte of each channel.
    #[test]
    fn test_r16g16b16a16_takes_high_bytes() {
        check_one_pixel(
            &[0x11, 0xAA,  0x22, 0xBB,  0x33, 0xCC,  0x44, 0xDD],
            Format::R16G16B16A16,
            [0xAA, 0xBB, 0xCC, 0xDD],
        );
    }

    /// R16G16B16 → RGBA8: high byte of each channel, alpha fixed at 255.
    #[test]
    fn test_r16g16b16_takes_high_bytes_adds_alpha() {
        check_one_pixel(
            &[0x11, 0xAA,  0x22, 0xBB,  0x33, 0xCC],
            Format::R16G16B16,
            [0xAA, 0xBB, 0xCC, 255],
        );
    }

    /// R16G16 → RGBA8: high byte of R and G; blue = 0, alpha = 255.
    #[test]
    fn test_r16g16_takes_high_bytes_pads_blue() {
        check_one_pixel(
            &[0x11, 0xAA,  0x22, 0xBB],
            Format::R16G16,
            [0xAA, 0xBB, 0, 255],
        );
    }

    /// R16 → RGBA8: high byte replicated into R, G, B; alpha = 255.
    #[test]
    fn test_r16_replicates_high_byte_to_grayscale() {
        check_one_pixel(&[0x11, 0xAA], Format::R16, [0xAA, 0xAA, 0xAA, 255]);
    }

    // ── Multi-pixel correctness ────────────────────────────────────────────
    //
    // Single-pixel tests verify the per-pixel formula; multi-pixel tests
    // verify that the chunking and output accumulation produce the right
    // sequence across an entire row.

    /// R8G8B8 over two pixels: ensure alpha is appended after every triplet,
    /// not just at the end.
    #[test]
    fn test_r8g8b8_two_pixels() {
        let out = to_rgba8(&[10, 20, 30,  40, 50, 60], Format::R8G8B8, 2, 1);
        assert_eq!(out, vec![10, 20, 30, 255,  40, 50, 60, 255]);
    }

    /// R8 grayscale over three pixels: each byte must become four output bytes.
    #[test]
    fn test_r8_three_pixels() {
        let out = to_rgba8(&[0, 127, 255], Format::R8, 3, 1);
        assert_eq!(out, vec![
            0,   0,   0,   255,
            127, 127, 127, 255,
            255, 255, 255, 255,
        ]);
    }

    /// R16G16B16A16 over two pixels: verify both pixels are decoded, not just
    /// the first (guards against an off-by-one in the chunk iterator).
    #[test]
    fn test_r16g16b16a16_two_pixels() {
        let out = to_rgba8(
            &[0x01, 0xAA, 0x02, 0xBB, 0x03, 0xCC, 0x04, 0xDD,   // pixel 1
              0x05, 0x11, 0x06, 0x22, 0x07, 0x33, 0x08, 0x44],  // pixel 2
            Format::R16G16B16A16, 2, 1,
        );
        assert_eq!(out, vec![0xAA, 0xBB, 0xCC, 0xDD,  0x11, 0x22, 0x33, 0x44]);
    }
}
