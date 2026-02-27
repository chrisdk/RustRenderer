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

        // HDR floating-point formats (32-bit per channel) are used for
        // environment maps and other high-dynamic-range content. These require
        // special handling that isn't implemented yet, so we return a solid
        // white placeholder for now.
        _ => vec![255u8; n * 4],
    }
}
