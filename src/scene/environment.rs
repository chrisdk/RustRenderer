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
    // Not all tools write the FORMAT line; tolerate its absence silently.
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
            #[allow(clippy::needless_range_loop)]  // c indexes the inner [u8;4], not rgbe_buf
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Builds a minimal well-formed HDR header for a given image size.
    ///
    /// The header ends with a blank line followed by the resolution string, as
    /// required by the Radiance format. Pixels follow immediately after.
    fn hdr_header(width: u32, height: u32) -> Vec<u8> {
        let mut h = Vec::new();
        h.extend_from_slice(b"#?RADIANCE\n");
        h.extend_from_slice(b"FORMAT=32-bit_rle_rgbe\n");
        h.extend_from_slice(b"\n");                              // blank line = end of header
        h.extend_from_slice(format!("-Y {height} +X {width}\n").as_bytes());
        h
    }

    /// Constructs a 1×1 uncompressed HDR containing a single RGBE pixel.
    ///
    /// Because the first two bytes of the pixel data are not [2, 2], the decoder
    /// takes the uncompressed branch (width=1 means scanline_width ≠ 1 anyway).
    fn single_pixel_hdr(rgbe: [u8; 4]) -> Vec<u8> {
        let mut data = hdr_header(1, 1);
        data.extend_from_slice(&rgbe);
        data
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// A zero exponent byte means "this pixel is black" — the formula
    /// `mantissa × 2^(E−128)` is not evaluated, avoiding 0 × 2^−128.
    #[test]
    fn test_decode_hdr_black_pixel() {
        let hdr = single_pixel_hdr([0, 0, 0, 0]);
        let img = decode_hdr(&hdr).expect("should decode without error");
        assert_eq!(img.width,  1);
        assert_eq!(img.height, 1);
        assert_eq!(img.pixels.len(), 4,  "one pixel = 4 f32s (R, G, B, padding)");
        assert_eq!(img.pixels[0], 0.0,  "R should be 0");
        assert_eq!(img.pixels[1], 0.0,  "G should be 0");
        assert_eq!(img.pixels[2], 0.0,  "B should be 0");
        assert_eq!(img.pixels[3], 1.0,  "padding channel is always 1.0");
    }

    /// RGBE pixel [128, 0, 0, 129]:
    ///   scale = 2^(129−128) / 256 = 2/256 = 1/128
    ///   R = 128 × (1/128) = 1.0, G = 0.0, B = 0.0
    #[test]
    fn test_decode_hdr_uncompressed_red() {
        let hdr = single_pixel_hdr([128, 0, 0, 129]);
        let img = decode_hdr(&hdr).expect("should decode");
        let eps = 1e-5f32;
        assert!((img.pixels[0] - 1.0).abs() < eps, "R expected 1.0, got {}", img.pixels[0]);
        assert!((img.pixels[1] -  0.0).abs() < eps, "G expected 0.0, got {}", img.pixels[1]);
        assert!((img.pixels[2] -  0.0).abs() < eps, "B expected 0.0, got {}", img.pixels[2]);
    }

    /// RGBE pixel [128, 64, 32, 129]:
    ///   scale = 1/128
    ///   R = 128/128 = 1.0, G = 64/128 = 0.5, B = 32/128 = 0.25
    #[test]
    fn test_decode_hdr_uncompressed_mixed_channels() {
        let hdr = single_pixel_hdr([128, 64, 32, 129]);
        let img = decode_hdr(&hdr).expect("should decode");
        let eps = 1e-5f32;
        assert!((img.pixels[0] - 1.00).abs() < eps, "R expected 1.00, got {}", img.pixels[0]);
        assert!((img.pixels[1] - 0.50).abs() < eps, "G expected 0.50, got {}", img.pixels[1]);
        assert!((img.pixels[2] - 0.25).abs() < eps, "B expected 0.25, got {}", img.pixels[2]);
    }

    /// New-RLE scanline: magic bytes [2, 2, 0, 2] signal a 2-pixel-wide RLE row.
    /// Each channel is encoded as a run of 2 identical bytes:
    ///   code 130 (> 128) → run of 130−128 = 2 repetitions of the following byte.
    /// Both pixels decode to RGBE [128, 64, 32, 129] = (1.0, 0.5, 0.25, 1.0).
    #[test]
    fn test_decode_hdr_new_rle_scanline() {
        let mut hdr = hdr_header(2, 1);
        // New-RLE magic header for a 2-pixel-wide scanline.
        hdr.extend_from_slice(&[2, 2, 0, 2]);
        // Four channel runs, each: [130=run-of-2, value].
        hdr.extend_from_slice(&[130, 128]);   // R channel: 128, 128
        hdr.extend_from_slice(&[130,  64]);   // G channel: 64, 64
        hdr.extend_from_slice(&[130,  32]);   // B channel: 32, 32
        hdr.extend_from_slice(&[130, 129]);   // E channel: 129, 129

        let img = decode_hdr(&hdr).expect("should decode RLE");
        assert_eq!(img.width,  2);
        assert_eq!(img.height, 1);
        assert_eq!(img.pixels.len(), 8, "2 pixels × 4 f32s = 8");

        let eps = 1e-5f32;
        for px in 0..2 {
            let base = px * 4;
            assert!((img.pixels[base    ] - 1.00).abs() < eps, "px{px} R");
            assert!((img.pixels[base + 1] - 0.50).abs() < eps, "px{px} G");
            assert!((img.pixels[base + 2] - 0.25).abs() < eps, "px{px} B");
            assert_eq!(img.pixels[base + 3], 1.0,                "px{px} padding");
        }
    }

    /// A file truncated before the resolution string should return a clear error,
    /// not panic or produce garbage output.
    #[test]
    fn test_decode_hdr_truncated_returns_error() {
        let result = decode_hdr(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n");
        assert!(result.is_err(), "truncated file should return Err");
    }

    /// Completely invalid bytes should return an error rather than panicking.
    #[test]
    fn test_decode_hdr_garbage_returns_error() {
        let result = decode_hdr(b"this is not an HDR file");
        assert!(result.is_err());
    }
}
