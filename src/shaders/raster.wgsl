// =============================================================================
// Rasterising preview shader — vertex + fragment
// =============================================================================
//
// This shader powers the *fast* interactive preview path. Path tracing is
// gorgeous but slow; rasterisation is approximate but runs at 60 fps. The
// goal here is "close enough to spin the model and get your bearings" rather
// than "physically accurate". The path tracer handles that when the user
// clicks Render.
//
// Algorithm: single-pass PBR approximation.
//   – All five texture maps applied: albedo (sRGB), normal, metallic/roughness,
//     emissive, and ambient occlusion — bilinear filtered.
//   – TBN normal mapping via stored vertex tangents (GLTF convention: tangent.w
//     gives handedness; bitangent = cross(N, T) * w).
//   – Directional sun (azimuth/elevation/intensity from per-frame uniform) +
//     diffuse IBL ambient + specular IBL (split-sum, Karis 2013) — HDR env map
//     when loaded, procedural sky gradient as fallback.
//   – Sky background pass (fullscreen triangle, depth_compare = Always) renders
//     the env map or procedural gradient behind all geometry.
//   – GGX NDF specular + Schlick Fresnel (no G2/V term — fast and close enough).
//   – No hard shadows, no refraction — the path tracer handles those.
//   – ACES filmic tonemap + sRGB gamma; all UI controls live-update the preview.
//
// The Material struct here must byte-for-byte match src/scene/material.rs.
// The TexInfo struct must match GpuTexInfo in src/renderer/raster.rs.
// =============================================================================


// =============================================================================
// Bind group 0
// =============================================================================

/// Camera matrices plus world-space position for specular highlights.
/// Rust: RasterCameraUniform (144 bytes).
struct RasterCamera {
    view:    mat4x4<f32>,  // world → camera space       (offset 0,   64 B)
    proj:    mat4x4<f32>,  // camera space → clip space  (offset 64,  64 B)
    cam_pos: vec3<f32>,    // world-space camera position (offset 128, 12 B)
    _pad:    f32,          // padding to 144 B            (offset 140)
}


/// Material parameters, byte-for-byte identical to src/scene/material.rs.
/// 64 bytes.
///
/// Note: `emissive` is a vec3<f32> at offset 16. WGSL storage-buffer layout
/// gives vec3<f32> AlignOf=16 and SizeOf=12, so it lands at offset 16 with
/// the next field (metallic) at offset 28 — matching the Rust layout exactly.
struct Material {
    albedo:                     vec4<f32>,   // base colour + alpha  (offset 0)
    emissive:                   vec3<f32>,   // self-emission        (offset 16)
    metallic:                   f32,         //                      (offset 28)
    roughness:                  f32,         //                      (offset 32)
    albedo_texture:             i32,         //                      (offset 36)
    normal_texture:             i32,         //                      (offset 40)
    metallic_roughness_texture: i32,         //                      (offset 44)
    emissive_texture:           i32,         //                      (offset 48)
    ior:                        f32,         //                      (offset 52)
    transmission:               f32,         //                      (offset 56)
    occlusion_texture:          i32,         //                      (offset 60)
}

/// Per-texture metadata. Must match GpuTexInfo in raster.rs (16 bytes).
struct TexInfo {
    offset: u32,  // pixel index into tex_data where this texture starts
    width:  u32,
    height: u32,
    _pad:   u32,
}

/// Per-frame lighting controls passed from Rust. Mirrors `RasterFrameUniforms`
/// in raster.rs — field offsets must match exactly.
///
/// Note: no explicit `_pad` field here even though the Rust struct has one.
/// WGSL automatically pads a struct to a multiple of its alignment. With
/// `env_background` (the last field) ending at byte 40 and AlignOf = 16,
/// the shader sees this struct as `roundUp(40, 16) = 48` bytes — matching
/// the Rust side. `array<u32, N>` would be illegal in `var<uniform>` because
/// WGSL requires array element alignment ≥ 16; plain u32 fields are safe.
struct RasterFrame {
    sun_dir:        vec3<f32>,  // normalised world-space direction toward the sun (offset 0)
    sun_intensity:  f32,        // radiance multiplier; 0 = off, 1 = default        (offset 12)
    exposure:       f32,        // EV stops; applied as linear × 2^exposure          (offset 16)
    ibl_scale:      f32,        // multiplier for all sky/env contributions           (offset 20)
    env_available:  u32,        // 1 = HDR env map loaded and IBL enabled            (offset 24)
    env_width:      u32,        // env map pixel width                               (offset 28)
    env_height:     u32,        // env map pixel height                              (offset 32)
    env_background: u32,        // 1 = draw env/sky as background (sky pass)         (offset 36)
    // struct ends at 40; AlignOf = 16 → SizeOf = roundUp(40, 16) = 48 bytes.
}

@group(0) @binding(0) var<uniform>       camera:     RasterCamera;
@group(0) @binding(2) var<storage, read> materials:  array<Material>;
/// Flat RGBA8 pixel data for all textures. One u32 per pixel: R in low byte,
/// A in high byte. Same packing as the path tracer's tex_data buffer.
@group(0) @binding(3) var<storage, read> tex_data:   array<u32>;
/// Per-texture metadata (pixel offset + dimensions). Indexed by material texture index.
@group(0) @binding(4) var<storage, read> tex_info:   array<TexInfo>;
/// Per-frame lighting controls (sun, exposure, IBL scale, env dimensions).
@group(0) @binding(5) var<uniform>       frame:      RasterFrame;
/// HDR environment map pixels as `vec4<f32>` (linear RGB, A unused).
/// Shared directly with the path tracer — no duplication. Always bound;
/// `frame.env_available` controls whether the shader actually samples it.
@group(0) @binding(6) var<storage, read> env_pixels: array<vec4<f32>>;


// =============================================================================
// Texture sampling
// =============================================================================

/// Unpacks a packed RGBA8 texel (one u32, R in low byte) into a linear vec4.
///
/// Mirrors `unpack_texel()` in trace.wgsl — both sides of the preview/render
/// split must decode pixels identically or colours will differ between modes.
fn unpack_rgba8(raw: u32) -> vec4<f32> {
    return vec4<f32>(
        f32( raw        & 0xFFu) / 255.0,
        f32((raw >>  8u) & 0xFFu) / 255.0,
        f32((raw >> 16u) & 0xFFu) / 255.0,
        f32((raw >> 24u) & 0xFFu) / 255.0,
    );
}

/// Samples a packed RGBA8 texture from the flat storage buffer using
/// **bilinear filtering** (4-tap blend), matching `sample_texture()` in
/// trace.wgsl.
///
/// Returns (1,1,1,1) for invalid or absent textures (idx < 0), so multiplying
/// by the result is always safe.
///
/// UV coordinates wrap with fract() (standard GLTF REPEAT mode). The -0.5 texel
/// offset aligns the sample point with texel *centres* rather than texel
/// *corners* — the same convention used by hardware sampler units and by the
/// path tracer, so both modes filter identically on the same texture data.
///
/// Bilinear filtering is especially important for normal maps: nearest-neighbour
/// on a low-resolution normal map produces visible faceting on curved surfaces
/// that vanishes after switching to the path tracer.
fn sample_tex(idx: i32, uv: vec2<f32>) -> vec4<f32> {
    if (idx < 0) { return vec4<f32>(1.0); }
    let info = tex_info[u32(idx)];
    let W = info.width;
    let H = info.height;
    if (W == 0u || H == 0u) { return vec4<f32>(1.0); }

    // Map UV to texel space with -0.5 offset for texel-centre alignment.
    let u = fract(uv.x) * f32(W) - 0.5;
    let v = fract(uv.y) * f32(H) - 0.5;

    let x0i = i32(floor(u));
    let y0i = i32(floor(v));

    // Clamp to valid range (GLTF CLAMP_TO_EDGE within each texel neighbourhood).
    let x0 = u32(clamp(x0i,     0, i32(W) - 1));
    let y0 = u32(clamp(y0i,     0, i32(H) - 1));
    let x1 = u32(clamp(x0i + 1, 0, i32(W) - 1));
    let y1 = u32(clamp(y0i + 1, 0, i32(H) - 1));

    // Sub-texel fractional weights.
    let fx = clamp(u - f32(x0i), 0.0, 1.0);
    let fy = clamp(v - f32(y0i), 0.0, 1.0);

    let p00 = unpack_rgba8(tex_data[info.offset + y0 * W + x0]);
    let p10 = unpack_rgba8(tex_data[info.offset + y0 * W + x1]);
    let p01 = unpack_rgba8(tex_data[info.offset + y1 * W + x0]);
    let p11 = unpack_rgba8(tex_data[info.offset + y1 * W + x1]);

    // Bilinear blend: lerp horizontally then vertically.
    return mix(mix(p00, p10, fx), mix(p01, p11, fx), fy);
}

/// Samples the HDR environment map in the direction `dir` using an
/// equirectangular (lat-long) mapping with bilinear filtering.
///
/// Mirrors `sample_env()` in trace.wgsl so preview and path-tracer ambient
/// colours match. Called only when `frame.env_available != 0`.
///
/// Convention (Poly Haven / most HDRI tools):
///   longitude = atan2(dir.x, −dir.z)  →  U = 0 at −Z (forward), increases right
///   latitude  = asin(dir.y)           →  V = 0 at top (+Y), 1 at bottom
fn sample_env(dir: vec3<f32>) -> vec3<f32> {
    let pi     = 3.14159265358979;
    let two_pi = 6.28318530717959;

    let phi   = atan2(dir.x, -dir.z);                 // [-π, π]
    let theta = asin(clamp(dir.y, -0.9999, 0.9999));  // [-π/2, π/2]

    let u = phi   / two_pi + 0.5;  // [0, 1]
    let v = 0.5   - theta  / pi;   // [0, 1], 0 = top

    let W = frame.env_width;
    let H = frame.env_height;

    // Texel-centre alignment (same -0.5 offset as sample_tex / trace.wgsl).
    let uf = u * f32(W) - 0.5;
    let vf = v * f32(H) - 0.5;

    let x0i = i32(floor(uf));
    let y0i = i32(floor(vf));

    // Wrap U (equirectangular is periodic east-west), clamp V (poles).
    let x0 = u32(((x0i     % i32(W)) + i32(W)) % i32(W));
    let x1 = u32((((x0i+1) % i32(W)) + i32(W)) % i32(W));
    let y0 = u32(clamp(y0i,     0, i32(H) - 1));
    let y1 = u32(clamp(y0i + 1, 0, i32(H) - 1));

    let fx = clamp(uf - f32(x0i), 0.0, 1.0);
    let fy = clamp(vf - f32(y0i), 0.0, 1.0);

    let p00 = env_pixels[y0 * W + x0].rgb;
    let p10 = env_pixels[y0 * W + x1].rgb;
    let p01 = env_pixels[y1 * W + x0].rgb;
    let p11 = env_pixels[y1 * W + x1].rgb;

    return mix(mix(p00, p10, fx), mix(p01, p11, fx), fy);
}


// =============================================================================
// Vertex shader
// =============================================================================

struct VertexInput {
    @location(0) position: vec3<f32>,  // offset 0  (from Vertex struct)
    @location(1) normal:   vec3<f32>,  // offset 12
    @location(2) uv:       vec2<f32>,  // offset 24
    @location(3) tangent:  vec4<f32>,  // offset 32 — xyz tangent dir, w handedness
}

/// Per-instance data fed as vertex attributes with step_mode = Instance.
///
/// Using vertex attributes (rather than a storage buffer indexed by
/// @builtin(instance_index)) avoids the `maxStorageBuffersInVertexStage`
/// limit, which Chrome/Dawn enforces as 0 — any vertex-stage storage buffer
/// silently causes the draw call to produce nothing on those browsers.
///
/// Each of the four matrix columns occupies one vec4 attribute slot (locations
/// 4–7). The material index is a u32 at location 8. The Rust side packs these
/// fields identically in `GpuInstance`, with the instance vertex buffer set to
/// step_mode = Instance so one `GpuInstance` row is consumed per draw call.
struct InstanceInput {
    @location(4) col0:      vec4<f32>,  // transform matrix column 0 (local → world)
    @location(5) col1:      vec4<f32>,  // transform matrix column 1
    @location(6) col2:      vec4<f32>,  // transform matrix column 2
    @location(7) col3:      vec4<f32>,  // transform matrix column 3
    @location(8) mat_index: u32,        // index into the materials storage buffer
}

struct VertexOutput {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0) world_pos:          vec3<f32>,
    @location(1) world_normal:       vec3<f32>,
    @location(2) uv:                 vec2<f32>,
    @location(3) @interpolate(flat) mat_index: u32,
    @location(4) world_tangent:      vec4<f32>,  // xyz = world-space tangent, w = handedness
}

@vertex
fn vs_main(in: VertexInput, inst: InstanceInput) -> VertexOutput {
    // Reconstruct the mat4x4 transform from the four column vectors delivered
    // as vertex attributes. WGSL mat4x4 constructor takes columns in order.
    let transform = mat4x4<f32>(inst.col0, inst.col1, inst.col2, inst.col3);

    // Transform position: local → world → camera → clip.
    let world_pos4 = transform * vec4<f32>(in.position, 1.0);
    let clip_pos   = camera.proj * camera.view * world_pos4;

    // Upper-left 3×3 of the model matrix, used to transform direction vectors.
    // For non-uniform scaling, the correct normal transform is the transpose of
    // the inverse, but for the preview pass this approximation looks fine.
    let m3 = mat3x3<f32>(
        transform[0].xyz,
        transform[1].xyz,
        transform[2].xyz,
    );
    let world_normal  = normalize(m3 * in.normal);
    // Tangent transforms the same way as normals for direction vectors.
    let world_tangent = vec4<f32>(normalize(m3 * in.tangent.xyz), in.tangent.w);

    var out: VertexOutput;
    out.clip_pos      = clip_pos;
    out.world_pos     = world_pos4.xyz;
    out.world_normal  = world_normal;
    out.uv            = in.uv;
    out.mat_index     = inst.mat_index;
    out.world_tangent = world_tangent;
    return out;
}


// =============================================================================
// Normal mapping
// =============================================================================

/// Applies a normal map to the interpolated surface normal using the stored
/// vertex tangent (GLTF TBN convention).
///
/// When no normal map is present (tex_idx < 0) or the map sample is degenerate,
/// the unmodified smooth normal is returned.
fn apply_normal_map(
    tex_idx: i32,
    uv:      vec2<f32>,
    N:       vec3<f32>,    // world-space smooth normal (already normalised)
    T4:      vec4<f32>,    // world-space tangent xyz + handedness w
) -> vec3<f32> {
    if (tex_idx < 0) { return N; }

    let ts = sample_tex(tex_idx, uv).xyz * 2.0 - 1.0;  // [0,1] → [-1,1]
    if (dot(ts, ts) < 0.01) { return N; }  // degenerate sample (blank normal map?)

    let T = normalize(T4.xyz);
    // Bitangent = cross(N, T) * handedness. The sign flip from handedness
    // ensures the TBN frame is right-handed regardless of UV mirroring.
    let B = normalize(cross(N, T)) * T4.w;

    // TBN transforms tangent-space → world-space.
    return normalize(T * ts.x + B * ts.y + N * ts.z);
}


// =============================================================================
// Lighting
// =============================================================================

// Sky gradient colours for the procedural hemispherical ambient fallback.
// Blend from warm ground to cool blue zenith based on the surface normal's Y.
const SKY_ZENITH:  vec3<f32> = vec3<f32>(0.25, 0.45, 0.75);  // blue sky
const SKY_HORIZON: vec3<f32> = vec3<f32>(0.55, 0.65, 0.75);  // pale horizon
const GROUND_COL:  vec3<f32> = vec3<f32>(0.15, 0.13, 0.10);  // warm dark ground

const PI: f32 = 3.14159265358979;

/// Procedural hemispherical sky ambient — varies the ambient colour based on
/// the surface normal direction. Surfaces facing up get blue sky; facing down
/// get warm ground; horizontal surfaces get the horizon colour.
///
/// This is a much better approximation of environment lighting than a flat
/// ambient constant, and costs one mix() call. Used as the fallback when no
/// HDR env map is loaded (or when IBL is disabled).
fn sky_ambient(N: vec3<f32>) -> vec3<f32> {
    let t = N.y * 0.5 + 0.5;  // maps [-1, 1] → [0, 1]: down=0, up=1
    let sky = mix(SKY_HORIZON, SKY_ZENITH, max(N.y, 0.0));
    return mix(GROUND_COL, sky, t);
}

/// Returns the ambient sky colour for a given surface normal direction.
///
/// When an HDR environment map is loaded and IBL is enabled, samples the real
/// env map at direction `N` — metallic surfaces will pick up the actual sky
/// colour from the HDR. Otherwise falls back to the procedural gradient, which
/// costs almost nothing and still looks good for quick interactive preview.
///
/// Either way, the result is scaled by `frame.ibl_scale` so the IBL-scale
/// slider in the UI affects the rasteriser preview identically to the path tracer.
fn ambient_color(N: vec3<f32>) -> vec3<f32> {
    if (frame.env_available != 0u) {
        return sample_env(N) * frame.ibl_scale;
    }
    return sky_ambient(N) * frame.ibl_scale;
}

/// Samples the environment in direction `dir`, blurred by averaging 9 points
/// in a 3×3 grid spread over a cone of half-angle `roughness * 2.0` radians.
///
/// The path tracer approximates the specular integral by averaging many GGX-
/// sampled directions across the lobe. At roughness=0.3 the GGX lobe spans
/// roughly 60–90° end-to-end; a single env-map sample at R reproduces a crisp
/// mirror reflection even on medium-rough metallic surfaces. The 3×3 grid with
/// a wide cone (34° half-angle at roughness=0.3) averages across sky, horizon
/// and terrain, giving the soft, blurred reflection the path tracer achieves
/// without needing a prefiltered env map.
///
/// For near-mirrors (roughness < 0.05) returns a single sharp sample — blurring
/// a delta-function lobe would be wrong. Falls back to sky_ambient when no HDR
/// env map is loaded.
fn blurred_specular_env(dir: vec3<f32>, roughness: f32) -> vec3<f32> {
    if (frame.env_available == 0u) {
        return sky_ambient(dir) * frame.ibl_scale;
    }
    let c0 = sample_env(dir);
    if (roughness < 0.05) { return c0 * frame.ibl_scale; }

    // Orthonormal tangent frame around `dir` for placing spread samples.
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(dir.y) < 0.9);
    let t1 = normalize(cross(up, dir));
    let t2 = cross(dir, t1);
    // Cardinal spread (axis-aligned) and diagonal spread (0.707 × axis).
    let s  = roughness * 2.0;
    let sd = s * 0.7071;
    // 3×3 grid: center + 4 cardinal + 4 diagonal = 9 samples.
    let avg = (c0
             + sample_env(normalize(dir + s  * t1))
             + sample_env(normalize(dir - s  * t1))
             + sample_env(normalize(dir + s  * t2))
             + sample_env(normalize(dir - s  * t2))
             + sample_env(normalize(dir + sd * t1 + sd * t2))
             + sample_env(normalize(dir - sd * t1 + sd * t2))
             + sample_env(normalize(dir + sd * t1 - sd * t2))
             + sample_env(normalize(dir - sd * t1 - sd * t2))) * (1.0 / 9.0);
    return avg * frame.ibl_scale;
}

/// ACES filmic tonemap (Narkowicz 2015 fit) followed by sRGB gamma encode.
///
/// Shared by the geometry fragment shader (`fs_main`) and the sky background
/// pass (`fs_sky`) so both pipelines produce identical colour for the same
/// input radiance — no seam visible at the geometry silhouette edge.
///
/// Reference: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
fn tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let mapped = clamp(
        (color * (a * color + b)) / (color * (c * color + d) + e),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    return pow(mapped, vec3<f32>(1.0 / 2.2));
}


// =============================================================================
// Fragment shader
// =============================================================================

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let mat = materials[in.mat_index];
    let uv  = in.uv;

    // ---- Texture lookups ----
    // sample_tex returns (1,1,1,1) for idx < 0, so multiplying is always safe.
    //
    // GLTF spec: albedo and emissive textures are sRGB-encoded (gamma ~2.2).
    // All lighting math must happen in linear light, so decode them here.
    // Metallic/roughness and normal maps are linear data — no decode needed.
    let albedo_samp  = pow(sample_tex(mat.albedo_texture,             uv).rgb, vec3<f32>(2.2));
    let mr_samp      = sample_tex(mat.metallic_roughness_texture, uv);
    let emit_samp    = pow(sample_tex(mat.emissive_texture,           uv).rgb, vec3<f32>(2.2));

    // ---- Material parameters ----
    var albedo    = mat.albedo.rgb * albedo_samp;
    var metallic  = mat.metallic  * mr_samp.b;    // GLTF: metallic  in blue channel
    var roughness = mat.roughness * mr_samp.g;    // GLTF: roughness in green channel
    roughness = max(roughness, 0.04);             // avoid division by zero in NDF

    // ---- Normal (with map applied) ----
    let N_base = normalize(in.world_normal);
    let N      = apply_normal_map(mat.normal_texture, uv, N_base, in.world_tangent);

    // ---- Lighting vectors ----
    // Sun direction comes from the per-frame uniform so the azimuth/elevation
    // sliders in the UI move the highlight in the preview just as they would
    // in a path-traced render.
    let L = normalize(frame.sun_dir);
    let V = normalize(camera.cam_pos - in.world_pos);
    let H = normalize(L + V);

    let n_dot_l = max(dot(N, L), 0.0);
    let n_dot_h = max(dot(N, H), 0.0);
    let h_dot_v = max(dot(H, V), 0.0);
    // n_dot_v: cosine of the view-to-normal angle. Used in the ambient Fresnel
    // term (split-sum IBL approximation: Karis 2013). Clamped to avoid division
    // artefacts at grazing angles.
    let n_dot_v = max(dot(N, V), 0.0);
    // Reflection vector: the direction from which the surface "sees" the environment.
    // Polished metals and dielectrics will sample the env map at this direction.
    let R = reflect(-V, N);

    // Warm-white sun colour scaled by the sun-intensity slider.
    // The base (4.0, 3.8, 3.0) matches the path tracer's SUN_RADIANCE constant,
    // so the two modes look similar at default settings.
    let sun_color = vec3<f32>(4.0, 3.8, 3.0) * frame.sun_intensity;

    // ---- Diffuse (Lambertian) ----
    // Metals have no diffuse response; their albedo colour goes into F0 instead.
    let diffuse = albedo * (1.0 - metallic) * n_dot_l * sun_color;

    // ---- Specular (GGX NDF + Schlick Fresnel) ----
    // We skip the geometric attenuation term (G2/Smith) for speed. The GGX
    // normal distribution function alone gives the right highlight shape.
    let alpha  = roughness * roughness;
    let alpha2 = alpha * alpha;
    let d_denom = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    let D = alpha2 / (PI * d_denom * d_denom);

    // Schlick Fresnel: F0 = 0.04 for dielectrics, albedo for metals.
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let F  = f0 + (1.0 - f0) * pow(1.0 - h_dot_v, 5.0);

    // Simple 1/4π denominator approximation (avoids the full Cook-Torrance
    // BRDF denominator — good enough for a preview pass).
    let specular = (D * F * n_dot_l * sun_color) * 0.25;

    // ---- Hemispherical ambient (diffuse + specular) ----
    // Occlusion texture (R channel): 0 = fully occluded, 1 = fully exposed.
    // Multiplying ambient by this darkens crevices and cavities where indirect
    // light cannot easily reach. Only ambient is affected — direct sun light is
    // not occluded by this baked term.
    let occlusion = sample_tex(mat.occlusion_texture, uv).r;

    // Sample the env map once at the surface normal (used for both diffuse ambient
    // and as the "rough limit" colour for specular ambient). Reusing it saves a
    // texture lookup.
    let env_N = ambient_color(N);

    // Diffuse ambient: Lambertian sky/env colour weighted by (1−metallic).
    // Metals have no diffuse response — their reflectance is all specular.
    let diffuse_ambient = albedo * env_N * (1.0 - metallic) * occlusion;

    // Specular ambient: Karis 2013 BRDF LUT amplitude × env colour,
    // with a two-pronged fix for the "everything looks chrome" problem.
    //
    // AMPLITUDE (Karis LUT): Raw Schlick Fresnel approaches 1.0 at grazing
    // angles regardless of roughness, making every surface look chrome-plated.
    // The LUT encodes ∫ f(l,v)·NdotL dl over the GGX lobe as (scale, bias):
    //   specular = F0·scale + bias
    // For rough surfaces, scale→0 and bias→0, correctly suppressing the
    // specular. This is an analytical fit; no texture lookup required.
    //
    // COLOUR — two complementary fixes:
    //
    // 1. Hard cutoff via `mirror_w`: surfaces with roughness ≥ 0.5 get zero
    //    directional reflection and see only env_N (the same smooth direction
    //    the diffuse term uses). mirror_w = pow(max(0, 1−roughness×2), 2) falls
    //    to zero at roughness=0.5, eliminating landscape reflections on matte
    //    and semi-rough surfaces entirely.
    //
    // 2. Cone blur via `blurred_specular_env`: for smoother surfaces where
    //    mirror_w > 0, the sample at R is replaced by an average of 5 directions
    //    in a cone of half-angle roughness×0.9 radians. This mimics the GGX
    //    lobe spread the path tracer achieves by sampling many microfacet
    //    directions. Without this blur, even 20% of a single sharp R sample can
    //    show a vivid sky-vs-terrain edge as a clear reflection band.
    let brdf_c0  = vec4<f32>(-1.0, -0.0275, -0.572,  0.022);
    let brdf_c1  = vec4<f32>( 1.0,  0.0425,  1.04,  -0.04);
    let brdf_r   = roughness * brdf_c0 + brdf_c1;
    let brdf_a   = min(brdf_r.x * brdf_r.x, exp2(-9.28 * n_dot_v)) * brdf_r.x + brdf_r.y;
    let brdf_ab  = vec2<f32>(-1.04, 1.04) * brdf_a + brdf_r.zw;
    // mirror_w: how much of the sharp reflection direction R contributes vs the
    // smooth normal direction N. Falls to zero at roughness=0.5 so anything
    // rougher sees only env_N — no directional landscape reflection at all.
    // Uses pow(…, 2) for a smooth roll-off rather than a hard cutoff.
    let mirror_w = pow(max(0.0, 1.0 - roughness * 2.0), 2.0);
    var specular_env = env_N;
    if (mirror_w > 0.005) {
        // blurred_specular_env averages 5 directions around R (cone ∝ roughness)
        // to approximate the GGX lobe spread; skip for rough surfaces where
        // mirror_w ≈ 0 anyway to avoid the 5× texture cost for no benefit.
        specular_env = mix(env_N, blurred_specular_env(R, roughness), mirror_w);
    }
    let specular_ambient = (f0 * brdf_ab.x + brdf_ab.y) * specular_env * occlusion;

    let ambient = diffuse_ambient + specular_ambient;

    // ---- Emissive ----
    let emissive = mat.emissive * emit_samp;

    // ---- Glass hint ----
    // Real refraction needs the path tracer. In preview, dim transmissive
    // surfaces so you can at least tell they're glassy.
    let opacity = 1.0 - mat.transmission * 0.7;

    var color = (diffuse + specular + ambient + emissive) * opacity;

    // ---- Exposure ----
    // Applied as linear × 2^stops, matching the path tracer's exposure formula.
    // Because this is a multiplicative pre-tone-map transform, changing the
    // exposure slider in the UI updates the preview immediately without
    // invalidating any cached state.
    color = color * pow(2.0, frame.exposure);

    // ACES filmic tonemap + sRGB gamma — via the shared tonemap() helper so
    // geometry and sky pass produce identical output for the same radiance.
    return vec4<f32>(tonemap(color), 1.0);
}


// =============================================================================
// Sky background pass
// =============================================================================
//
// A fullscreen triangle drawn *before* scene geometry in the same render pass.
// The sky pipeline uses:
//   depth_compare    = Always          → sky draws regardless of depth buffer
//   depth_write_enabled = false        → doesn't block geometry drawn after it
//
// Geometry drawn afterwards uses depth_compare = Less + depth_write = true, so
// it overwrites sky pixels wherever actual geometry exists. The result: the sky
// fills every pixel that no triangle covers, without an extra render pass.
//
// Sky direction reconstruction:
//   Each fragment has an NDC coordinate (nx, ny) ∈ [-1, 1]².
//   For a perspective projection with x-scale proj[0][0] and y-scale proj[1][1]:
//     view_dir = normalize(nx / proj[0][0], ny / proj[1][1], -1)
//   The camera looks along -Z in view space; proj[0][0] = f/aspect,
//   proj[1][1] = f where f = 1/tan(fov/2).
//   The upper-left 3×3 of the view matrix (view[col].xyz) is the rotation that
//   maps world → camera space. Its transpose (= inverse for orthonormal) gives
//   camera → world, so: world_dir = normalize(transpose(rot3x3) * view_dir).

struct SkyOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       ndc:      vec2<f32>,
}

/// Generates three vertices covering the entire NDC square with one triangle.
///
/// The giant-triangle trick avoids the T-junction seam you'd get from two
/// screen-aligned triangles. A single triangle with corners at (−1,−1),
/// (3,−1), (−1, 3) covers every pixel in [−1,1]² exactly once.
@vertex
fn vs_sky(@builtin(vertex_index) vid: u32) -> SkyOutput {
    // vid 0 → (−1, −1), vid 1 → (3, −1), vid 2 → (−1, 3)
    let x = f32(vid & 1u) * 4.0 - 1.0;
    let y = f32(vid >> 1u) * 4.0 - 1.0;
    var out: SkyOutput;
    // z = 1 puts the sky at the far plane in NDC (clip_pos.z / clip_pos.w = 1).
    // Geometry rendered afterwards has z < 1 and passes the Less depth test,
    // overwriting sky pixels. The sky never writes to the depth buffer so it
    // cannot block later geometry.
    out.clip_pos = vec4<f32>(x, y, 1.0, 1.0);
    out.ndc      = vec2<f32>(x, y);
    return out;
}

/// Samples the sky/env-map colour for the fragment's screen position and
/// applies exposure + ACES tonemap, matching the geometry pass output exactly.
@fragment
fn fs_sky(in: SkyOutput) -> @location(0) vec4<f32> {
    // ── Reconstruct world-space ray direction ────────────────────────────────
    // Unproject NDC (nx, ny) through the perspective projection to view space,
    // then rotate to world space. camera.proj[col][row] in column-major WGSL:
    //   proj[0][0] = f/aspect  (x-axis scale)
    //   proj[1][1] = f         (y-axis scale)
    let view_dir = normalize(vec3<f32>(
        in.ndc.x / camera.proj[0][0],
        in.ndc.y / camera.proj[1][1],
        -1.0,
    ));

    // Upper-left 3×3 of the view matrix: maps world → camera space.
    // Transpose (= inverse for orthonormal rotation): maps camera → world.
    let rot = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz,
    );
    let world_dir = normalize(transpose(rot) * view_dir);

    // ── Sky colour ───────────────────────────────────────────────────────────
    // Show the HDR env map when it's loaded, IBL is on, and env_background is
    // set. Otherwise fall back to the procedural gradient — this handles both
    // "no env map loaded" and "studio mode" (env lights scene but isn't the
    // backdrop). Either way, ibl_scale lets the user dim or brighten the sky.
    var sky: vec3<f32>;
    if (frame.env_available != 0u && frame.env_background != 0u) {
        sky = sample_env(world_dir) * frame.ibl_scale;
    } else {
        sky = sky_ambient(world_dir) * frame.ibl_scale;
    }

    // Apply exposure and tonemap — same pipeline as fs_main so geometry and
    // sky colours are consistent at every exposure setting.
    return vec4<f32>(tonemap(sky * pow(2.0, frame.exposure)), 1.0);
}
