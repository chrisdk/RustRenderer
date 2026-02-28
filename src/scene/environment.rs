use super::texture::Texture;

/// The background and ambient lighting of the scene, provided by an HDRI image.
///
/// An HDRI (High Dynamic Range Image) environment map is a 360° panoramic
/// photograph of a real or synthetic environment — a sky, a studio, an outdoor
/// scene — stored with enough brightness range to capture both dim shadows and
/// intense highlights. In path tracing, rays that escape the scene geometry
/// without hitting anything are aimed at this panorama to sample the incoming
/// light from that direction.
///
/// This gives scenes realistic ambient lighting and reflections "for free":
/// a shiny metal ball will reflect the sky, indoor objects will be softly lit
/// by a distant window, and so on, all driven by the single HDRI image.
pub struct Environment {
    /// The HDRI panorama. If None, the renderer falls back to a solid background
    /// color derived from the tint.
    pub texture: Option<Texture>,

    /// A color multiplier applied to all environment light. Defaults to white
    /// [1, 1, 1] (no change). Can be used to dim the environment or shift its
    /// color temperature without replacing the texture.
    pub tint: [f32; 3],
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            texture: None,
            tint:    [1.0, 1.0, 1.0],
        }
    }
}

// =============================================================================
// HDR file decoding
// =============================================================================

/// Decoded environment map: linear-light RGB32F pixels in row-major order.
///
/// Each pixel is stored as four `f32`s — R, G, B, and a padding 1.0 — so the
/// data can be uploaded directly as `array<vec4<f32>>` to a WGSL storage buffer.
pub struct HdrImage {
    pub width:  u32,
    pub height: u32,
    /// `width * height * 4` f32 values: [R, G, B, 1.0] per pixel, row-major.
    pub pixels: Vec<f32>,
}

/// Decodes a Radiance RGBE (`.hdr`) file into linear-light HDR pixels.
///
/// ## Format overview
///
/// Radiance HDR files pack high-dynamic-range colours into 4 bytes per pixel
/// using a shared exponent (RGBE encoding):
///   - bytes R, G, B store the mantissa of each channel / 256
///   - byte E stores the shared exponent biased by 128
///   - decoded value: channel = R_or_G_or_B / 256 × 2^(E − 128)
///
/// Modern files use per-scanline run-length encoding (new RLE) to reduce file
/// size. Each scanline starts with the magic bytes `[2, 2, w_hi, w_lo]`.
///
/// This decoder handles the new-RLE format produced by every major HDRI tool
/// (Blender, Photoshop, hdrihaven/Poly Haven, etc.). Old-format uncompressed
/// files are also handled. The rare old-RLE format returns an error.
pub fn decode_hdr(bytes: &[u8]) -> Result<HdrImage, String> {
    let mut pos = 0usize;

    // ── Parse ASCII header ────────────────────────────────────────────────────
    // The header is a series of `\n`-terminated lines. A blank line signals
    // the end of the header. Everything after is pixel data.
    let mut found_format = false;
    loop {
        let line_end = bytes[pos..].iter().position(|&b| b == b'\n')
            .map(|p| pos + p)
            .ok_or("HDR: unexpected end of header")?;
        let line = &bytes[pos..line_end];
        pos = line_end + 1;

        if line.starts_with(b"FORMAT=32-bit_rle_rgbe") || line.starts_with(b"FORMAT=32-bit_rle_xyze") {
            found_format = true;
        }
        if line.is_empty() {
            break;  // blank line = end of header
        }
    }
    // Not all tools write the FORMAT line; warn but continue.
    let _ = found_format;

    // ── Resolution string ─────────────────────────────────────────────────────
    // Format: "-Y <height> +X <width>" (most common) or "+Y/-X" variants.
    let res_end = bytes[pos..].iter().position(|&b| b == b'\n')
        .map(|p| pos + p)
        .ok_or("HDR: missing resolution string")?;
    let res = std::str::from_utf8(&bytes[pos..res_end])
        .map_err(|_| "HDR: non-UTF8 resolution string")?;
    pos = res_end + 1;

    let parts: Vec<&str> = res.split_whitespace().collect();
    if parts.len() < 4 {
        return Err(format!("HDR: malformed resolution '{res}'"));
    }
    let height = parts[1].parse::<u32>().map_err(|_| "HDR: bad height")?;
    let width  = parts[3].parse::<u32>().map_err(|_| "HDR: bad width")?;

    // ── Pixel data ────────────────────────────────────────────────────────────
    let n_pixels = (width * height) as usize;
    let mut rgbe_buf = vec![[0u8; 4]; n_pixels];
    let mut out_row = 0usize;

    while out_row < height as usize {
        let row_start = out_row * width as usize;

        // Peek at the first 4 bytes to determine encoding.
        if pos + 4 > bytes.len() {
            return Err("HDR: truncated before scanline".into());
        }
        let magic  = [bytes[pos], bytes[pos+1], bytes[pos+2], bytes[pos+3]];
        let scanline_width = (magic[2] as u32) * 256 + magic[3] as u32;

        if magic[0] == 2 && magic[1] == 2 && scanline_width == width {
            // ── New RLE: four separate channel runs ───────────────────────────
            pos += 4;  // consume magic header
            for c in 0..4usize {
                let mut x = 0usize;
                while x < width as usize {
                    if pos >= bytes.len() {
                        return Err("HDR: truncated in RLE channel".into());
                    }
                    let code = bytes[pos] as usize;
                    pos += 1;
                    if code > 128 {
                        // Run: repeat the next byte (code − 128) times.
                        let count = code - 128;
                        if pos >= bytes.len() { return Err("HDR: truncated run".into()); }
                        let val = bytes[pos]; pos += 1;
                        for _ in 0..count {
                            rgbe_buf[row_start + x][c] = val;
                            x += 1;
                        }
                    } else {
                        // Literal: copy the next `code` bytes verbatim.
                        for _ in 0..code {
                            if pos >= bytes.len() { return Err("HDR: truncated literal".into()); }
                            rgbe_buf[row_start + x][c] = bytes[pos];
                            pos += 1;
                            x += 1;
                        }
                    }
                }
            }
        } else if magic[0] == 1 && magic[1] == 1 && magic[2] == 1 {
            return Err("HDR: old-RLE format is not supported — re-save the file in a modern tool".into());
        } else {
            // ── Uncompressed: raw RGBE pixels ─────────────────────────────────
            for x in 0..width as usize {
                if pos + 4 > bytes.len() { return Err("HDR: truncated pixel".into()); }
                rgbe_buf[row_start + x] = [bytes[pos], bytes[pos+1], bytes[pos+2], bytes[pos+3]];
                pos += 4;
            }
        }
        out_row += 1;
    }

    // ── RGBE → linear-light f32 ───────────────────────────────────────────────
    // When E == 0 the pixel is black (avoids undefined 2^(0-128)/256 behaviour).
    // Otherwise: channel = mantissa / 256 × 2^(E − 128).
    let mut pixels = Vec::with_capacity(n_pixels * 4);
    for [r, g, b, e] in &rgbe_buf {
        if *e == 0 {
            pixels.extend_from_slice(&[0.0f32, 0.0f32, 0.0f32, 1.0f32]);
        } else {
            let scale = f32::exp2(*e as f32 - 128.0) / 256.0;
            pixels.extend_from_slice(&[
                *r as f32 * scale,
                *g as f32 * scale,
                *b as f32 * scale,
                1.0f32,
            ]);
        }
    }

    Ok(HdrImage { width, height, pixels })
}
