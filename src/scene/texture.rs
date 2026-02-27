use gltf::image::Format;

pub struct Texture {
    pub width:  u32,
    pub height: u32,
    pub data:   Vec<u8>, // always RGBA8
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

fn to_rgba8(pixels: &[u8], format: Format, width: u32, height: u32) -> Vec<u8> {
    let n = (width * height) as usize;
    match format {
        Format::R8G8B8A8 => pixels.to_vec(),
        Format::R8G8B8 => {
            let mut out = Vec::with_capacity(n * 4);
            for rgb in pixels.chunks_exact(3) {
                out.extend_from_slice(rgb);
                out.push(255);
            }
            out
        }
        Format::R8G8 => {
            let mut out = Vec::with_capacity(n * 4);
            for rg in pixels.chunks_exact(2) {
                out.extend_from_slice(&[rg[0], rg[1], 0, 255]);
            }
            out
        }
        Format::R8 => {
            let mut out = Vec::with_capacity(n * 4);
            for &r in pixels {
                out.extend_from_slice(&[r, r, r, 255]);
            }
            out
        }
        // 16-bit formats: downsample by taking the high byte of each channel.
        // Values are stored little-endian, so index 1 is the most significant byte.
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
        // HDR float formats (e.g. environment maps) are not handled here yet.
        _ => vec![255u8; n * 4],
    }
}
