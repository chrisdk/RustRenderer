// Path-tracing compute shader — physically-based, multi-bounce.
//
// One GPU thread per pixel. For each sample it:
//   1. Shoots a jittered primary ray from the camera.
//   2. Finds the closest triangle via BVH traversal.
//   3. At the hit point, evaluates emission, then stochastically chooses a
//      scatter event (diffuse, GGX specular, or glass refraction) weighted
//      by the surface BRDF and the Fresnel equations.
//   4. Spawns the scattered ray and repeats up to frame.max_bounces times.
//   5. Accumulates the result into a running-average float buffer.
//
// The quality improves with sample count: invoke with sample_index 0, 1, 2, …
// to progressively refine the image. The output buffer holds the average after
// each dispatch, so the canvas can be updated live while rendering.
//
// Physical model:
//   - Opaque surfaces: GGX microfacet BRDF with Schlick Fresnel and Smith G
//     (the standard PBR metallic/roughness workflow used by GLTF).
//   - Glass/transmissive surfaces: Snell's law refraction with exact Fresnel
//     (dielectric equations, not Schlick — glass needs more precision than
//     the approximation gives near the critical angle).
//   - Normal maps: stored per-vertex GLTF tangents (Mikktspace), giving the
//     same tangent basis the map was baked against.
//   - Sky: procedural gradient with a physically-sized sun disk.
//   - Tone mapping: ACES filmic curve (Narkowicz 2015 fit).
//
// Binding layout
// ──────────────
// Group 0 — scene data (static, uploaded once per loaded scene)
//   binding 0 : array<BvhNode>     — the flat BVH tree
//   binding 1 : array<BvhTriangle> — world-space triangles in BVH leaf order
//   binding 2 : array<Material>    — PBR materials indexed by triangle
//   binding 3 : array<u32>         — flat RGBA8 pixel data for all textures
//   binding 4 : array<TexInfo>     — per-texture offset, width, height
//   binding 5 : array<vec4<f32>>   — HDR env map pixels (equirectangular, linear)
//   binding 6 : array<f32>         — env marginal CDF over rows
//   binding 7 : array<f32>         — env conditional CDFs (row-major, W entries/row)
// Group 1 — per-frame data (updated every dispatch)
//   binding 0 : Camera uniform     — camera position + pre-scaled basis
//   binding 1 : FrameUniforms      — dimensions, sample index, flags, env info,
//                                    and user lighting controls (bounces, sun, IBL, exposure)
//   binding 2 : array<u32>         — output RGBA8 pixels, row-major
//   binding 3 : array<vec4<f32>>   — float accumulator, one vec4 per pixel

// ============================================================================
// Structs — byte layouts MUST match the corresponding Rust #[repr(C)] types.
// ============================================================================

// Matches accel::bvh::BvhNode (32 bytes).
struct BvhNode {
    aabb_min:       vec3<f32>,
    right_or_first: u32,
    aabb_max:       vec3<f32>,
    count:          u32,
}

// Matches accel::bvh::BvhTriangle (160 bytes).
//
// WGSL vec3<f32> has 16-byte alignment which would drift from the packed Rust
// [f32; 3] layout, so all fields are spelled out as individual scalars.
//
//   offsets   0..35  : v0/v1/v2 positions     (nine  f32s)
//   offsets  36..51  : i0, i1, i2, material   (four  u32s)
//   offsets  52..87  : n0/n1/n2 normals        (nine  f32s)
//   offsets  88..111 : uv0/uv1/uv2             (six   f32s)
//   offsets 112..159 : t0/t1/t2 tangents       (twelve f32s, xyz+w each)
struct BvhTriangle {
    v0x: f32, v0y: f32, v0z: f32,
    v1x: f32, v1y: f32, v1z: f32,
    v2x: f32, v2y: f32, v2z: f32,
    i0: u32, i1: u32, i2: u32,
    material_index: u32,
    n0x: f32, n0y: f32, n0z: f32,
    n1x: f32, n1y: f32, n1z: f32,
    n2x: f32, n2y: f32, n2z: f32,
    uv0u: f32, uv0v: f32,
    uv1u: f32, uv1v: f32,
    uv2u: f32, uv2v: f32,
    // xyz = world-space tangent direction, w = handedness (+1 or -1).
    // Stored from the GLTF file so normal maps decode against the same
    // tangent basis they were baked in.
    t0x: f32, t0y: f32, t0z: f32, t0w: f32,
    t1x: f32, t1y: f32, t1z: f32, t1w: f32,
    t2x: f32, t2y: f32, t2z: f32, t2w: f32,
}

// Matches scene::material::Material (64 bytes).
//
//   offset  0 : albedo        (vec4<f32>, 16 bytes)
//   offset 16 : emissive_r/g/b, metallic  (four f32s)
//   offset 32 : roughness, albedo_tex, normal_tex, mr_tex  (f32 + three i32s)
//   offset 48 : emissive_tex, ior, transmission, occlusion_tex  (i32 + two f32s + i32)
struct Material {
    albedo:       vec4<f32>,
    emissive_r:   f32,
    emissive_g:   f32,
    emissive_b:   f32,
    metallic:     f32,
    roughness:    f32,
    albedo_tex:   i32,
    normal_tex:   i32,
    mr_tex:       i32,
    emissive_tex:    i32,
    ior:             f32,
    transmission:    f32,
    // Baked ambient occlusion. Not applied here — the path tracer physically
    // simulates indirect light occlusion through bounce rays.
    occlusion_tex:   i32,
}

// Matches renderer::gpu::GpuTextureInfo (16 bytes).
struct TexInfo {
    offset: u32,
    width:  u32,
    height: u32,
    _pad:   u32,
}

// Matches camera::CameraUniform (64 bytes).
struct Camera {
    position:   vec3<f32>, _pad0: f32,
    fwd:        vec3<f32>, _pad1: f32,
    rgt_scaled: vec3<f32>, _pad2: f32,
    up_scaled:  vec3<f32>, _pad3: f32,
}

// Matches renderer::gpu::FrameUniforms (64 bytes).
struct FrameUniforms {
    width:        u32,
    height:       u32,
    sample_index: u32,
    flags:        u32,  // bit 0: preview mode (cap bounces to PREVIEW_BOUNCES)
                        // bit 1: IBL disabled (use procedural sky even if env is loaded)
    // Environment map dimensions. Both zero when no env map is loaded, which
    // causes sky_color() to fall back to the procedural gradient.
    env_width:    u32,
    env_height:   u32,
    // 1 = show env map / sky as background; 0 = dark grey for missed rays.
    // With MIS, escaped secondary rays only contribute the env if env_background
    // == 1; env NEE at surface hits still provides IBL regardless of this flag.
    env_background:    u32,
    // MIS normalisation constant: W·H / (total_sin_weighted_lum · 2π²).
    // pdf_solid_angle(dir) = luminance(env_sample(dir)) * env_sample_weight.
    // Zero when no env map is loaded — disables env NEE and MIS in the shader.
    env_sample_weight: f32,

    // ── User-adjustable lighting controls (offsets 32–55) ───────────────────
    max_bounces:   u32,  // path length cap; preview mode overrides with PREVIEW_BOUNCES
    sun_azimuth:   f32,  // radians: horizontal angle from +Z toward +X
    sun_elevation: f32,  // radians: vertical angle above horizon
    sun_intensity: f32,  // multiplier for SUN_RADIANCE in NEE; 0 = no sun
    ibl_scale:     f32,  // multiplier for all env-map / sky contributions
    exposure:      f32,  // EV stops: linear *= 2^exposure, applied before ACES

    // ── Depth of field (offsets 56–63) ───────────────────────────────────────
    aperture:   f32,  // thin-lens radius in world units; 0 = pinhole (no DoF)
    focus_dist: f32,  // focal plane distance in world units
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read>       bvh_nodes:  array<BvhNode>;
@group(0) @binding(1) var<storage, read>       bvh_tris:   array<BvhTriangle>;
@group(0) @binding(2) var<storage, read>       materials:  array<Material>;
@group(0) @binding(3) var<storage, read>       tex_data:   array<u32>;
@group(0) @binding(4) var<storage, read>       tex_info:   array<TexInfo>;
/// HDR environment map pixels as linear-light vec4<f32> (A component unused).
/// Row-major, frame.env_width × frame.env_height. One black pixel when not loaded.
@group(0) @binding(5) var<storage, read>       env_pixels:         array<vec4<f32>>;
/// Marginal CDF over env map rows: env_marginal_cdf[y] = P(row ≤ y).
/// Used for the first stage of 2-D importance sampling (choose a row).
/// One f32 per row; length = frame.env_height when loaded, 1 otherwise.
@group(0) @binding(6) var<storage, read>       env_marginal_cdf:    array<f32>;
/// Per-row conditional CDFs: env_conditional_cdf[y*W+x] = P(col ≤ x | row=y).
/// Used for the second stage (choose a column given the row).
/// Length = frame.env_width × frame.env_height when loaded, 1 otherwise.
@group(0) @binding(7) var<storage, read>       env_conditional_cdf: array<f32>;

@group(1) @binding(0) var<uniform>             camera:  Camera;
@group(1) @binding(1) var<uniform>             frame:   FrameUniforms;
@group(1) @binding(2) var<storage, read_write> output:  array<u32>;
@group(1) @binding(3) var<storage, read_write> accum:   array<vec4<f32>>;

// ============================================================================
// Constants
// ============================================================================

const T_MIN: f32 = 0.0001;     // minimum hit distance; avoids self-intersection
const T_MAX: f32 = 1.0e30;     // effective far plane

const PI:      f32 = 3.14159265358979323846;
const TWO_PI:  f32 = 6.28318530717958647692;
const INV_PI:  f32 = 0.31830988618379067154;

// Bounce cap for preview/drag mode. 2 bounces = direct light + one reflection
// or transmission. Cheap enough to stay interactive; still shows PBR materials.
// Full-quality bounce count is now driven by frame.max_bounces (default 12).
const PREVIEW_BOUNCES: i32 = 2;

// After this bounce index, Russian roulette may terminate the path.
const RR_START: i32 = 3;

// Sky colours at horizon and zenith, in linear light.
//
// These are calibrated for albedo textures that have been correctly decoded
// from sRGB to linear (pow(x, 2.2)). Linear albedo values are ~2–4× darker
// than raw sRGB, so the sky must be proportionally dimmer to avoid the bright
// ambient washing out surface colours.
//
// In a path tracer the sky contributes at every bounce, so it must be much
// dimmer than it would be in a rasteriser (where ambient is applied once).
// Target: ~40% display-brightness background after ACES + gamma (~0.10 linear).
const SKY_HORIZON: vec3<f32> = vec3<f32>(0.10, 0.09, 0.07);
const SKY_ZENITH:  vec3<f32> = vec3<f32>(0.03, 0.06, 0.15);

// Sun radiance. With NEE, every surface bounce fires a shadow ray toward the
// sun and computes the full BRDF contribution deterministically, so this value
// directly controls how bright direct sunlit surfaces appear.
const SUN_RADIANCE: vec3<f32> = vec3<f32>(10.0, 9.0, 7.0);

// ============================================================================
// PCG random number generator
// ============================================================================

// PCG-RXS-M-XS: minimal but high-quality 32-bit hash. Much better than a
// plain LCG — produces no visible patterns even in a path tracer where the
// sample count is low. Reference: O'Neill 2014.
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Advance the seed and return a uniform float in [0, 1).
fn rand_f32(seed: ptr<function, u32>) -> f32 {
    *seed = pcg(*seed);
    return f32(*seed) / 4294967296.0;
}

// Return two independent uniform samples in [0,1)².
fn rand2(seed: ptr<function, u32>) -> vec2<f32> {
    return vec2(rand_f32(seed), rand_f32(seed));
}

// ============================================================================
// Texture sampling
// ============================================================================

// Unpack a single RGBA8 texel stored as a packed u32 into a linear vec4.
fn unpack_texel(packed: u32) -> vec4<f32> {
    return vec4(
        f32( packed        & 0xFFu) / 255.0,
        f32((packed >>  8u) & 0xFFu) / 255.0,
        f32((packed >> 16u) & 0xFFu) / 255.0,
        f32((packed >> 24u) & 0xFFu) / 255.0,
    );
}

// Sample a texture by index, UV-wrapping, bilinear interpolation.
// tex_idx < 0 means "no texture" — returns white so the caller can multiply
// it against the material's scalar without changing the result.
//
// Bilinear filtering blends the four nearest texels, which smooths out normal
// maps viewed at oblique angles and removes the nearest-neighbour grain that
// was especially visible on curved metallic surfaces.
fn sample_texture(tex_idx: i32, uv: vec2<f32>) -> vec4<f32> {
    if tex_idx < 0 {
        return vec4(1.0, 1.0, 1.0, 1.0);
    }

    let info = tex_info[u32(tex_idx)];
    let W = info.width;
    let H = info.height;

    // Map UV to texel space, offset by -0.5 so texel centres sit at
    // integer coordinates (standard bilinear convention).
    let u = fract(uv.x) * f32(W) - 0.5;
    let v = fract(uv.y) * f32(H) - 0.5;

    // Integer texel indices for the two-by-two neighbourhood.
    let x0i = i32(floor(u));
    let y0i = i32(floor(v));
    let x0 = u32(clamp(x0i,     0, i32(W) - 1));
    let y0 = u32(clamp(y0i,     0, i32(H) - 1));
    let x1 = u32(clamp(x0i + 1, 0, i32(W) - 1));
    let y1 = u32(clamp(y0i + 1, 0, i32(H) - 1));

    // Sub-texel fractional position (clamped to [0,1] for safety).
    let fx = clamp(u - f32(x0i), 0.0, 1.0);
    let fy = clamp(v - f32(y0i), 0.0, 1.0);

    let p00 = unpack_texel(tex_data[info.offset + y0 * W + x0]);
    let p10 = unpack_texel(tex_data[info.offset + y0 * W + x1]);
    let p01 = unpack_texel(tex_data[info.offset + y1 * W + x0]);
    let p11 = unpack_texel(tex_data[info.offset + y1 * W + x1]);

    return mix(mix(p00, p10, fx), mix(p01, p11, fx), fy);
}

// ============================================================================
// Orthonormal basis (ONB) construction
// ============================================================================

// Build a tangent frame where `n` is the Z axis (normal direction).
// Uses the Duff et al. 2017 method ("Building an Orthonormal Basis, Revisited")
// which is numerically stable for all surface normals.
struct Onb { t: vec3<f32>, b: vec3<f32> }

fn make_onb(n: vec3<f32>) -> Onb {
    // The sign trick avoids the discontinuity at n.z = 0 that plagued earlier
    // methods. The math is a normalised Householder reflection. Trust the paper.
    let s  = select(-1.0, 1.0, n.z >= 0.0);
    let a  = -1.0 / (s + n.z);
    let b_ = n.x * n.y * a;
    return Onb(
        vec3(1.0 + s * n.x * n.x * a,  s * b_,         -s * n.x),
        vec3(b_,                         s + n.y*n.y*a,  -n.y),
    );
}

// Transform a direction from the ONB's local frame (Z=normal) to world space.
fn onb_to_world(onb: Onb, n: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return v.x * onb.t + v.y * onb.b + v.z * n;
}

// ============================================================================
// Hemisphere and GGX sampling
// ============================================================================

// Cosine-weighted hemisphere sample. Returns a direction in the ONB's local
// frame where Z = 1 is the normal. PDF = cos(theta) / PI.
fn cosine_hemisphere(xi: vec2<f32>) -> vec3<f32> {
    let phi       = TWO_PI * xi.x;
    let sin_theta = sqrt(xi.y);          // sin(theta) from the cosine-weighted CDF
    let cos_theta = sqrt(1.0 - xi.y);
    return vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

// Sample the Visible Normal Distribution Function (VNDF) for isotropic GGX.
//
// The old NDF sampler drew H from D(H)·(N·H), which is unbiased but wasteful:
// it generates many microsurface normals that are geometrically masked from the
// viewer's direction and contribute nothing. For a smooth metal at an oblique
// angle, the vast majority of NDF samples are wasted — hence the grain on
// chrome trim at 512 spp.
//
// VNDF conditions the distribution on the view direction, concentrating all
// samples on microsurfaces that are *actually visible* from V. The result is
// 4–8× lower variance on smooth metals with no change in the expected value.
//
// Reference: Heitz 2018, "Sampling the GGX Distribution of Visible Normals",
//            Journal of Computer Graphics Techniques vol.7 no.4.
//
// Parameters:
//   v_local  View direction in the ONB's local frame (z = surface normal).
//   alpha    Linearised roughness = perceptual_roughness².
//   xi       Two uniform random numbers in [0,1)².
// Returns:
//   The sampled half-vector H in local space.
//   PDF(L) = D(H) · G1(N·V) / (4 · N·V)   [reflected direction L = reflect(-V, H)]
fn ggx_vndf(v_local: vec3<f32>, alpha: f32, xi: vec2<f32>) -> vec3<f32> {
    // Step 1: stretch view into the "unit hemisphere" space where the anisotropic
    // GGX distribution looks like an ordinary hemisphere (Heitz §3.2).
    let Vh = normalize(vec3(alpha * v_local.x, alpha * v_local.y, v_local.z));

    // Step 2: orthonormal basis around Vh for sampling the projected disk.
    // Divide by sqrt(max(lensq, ε)) rather than lensq to avoid 0/0 when the
    // view is nearly along the normal (Vh.x ≈ Vh.y ≈ 0). The select() chooses
    // the safe (1,0,0) fallback; the unsafe arm evaluates to a finite non-NaN
    // value (small numerators / sqrt(ε)) which gets discarded.
    let lensq   = Vh.x * Vh.x + Vh.y * Vh.y;
    let inv_len = 1.0 / sqrt(max(lensq, 1e-8));
    let T1 = select(vec3(1.0, 0.0, 0.0),
                    vec3(-Vh.y * inv_len, Vh.x * inv_len, 0.0),
                    lensq > 1e-8);
    let T2 = cross(Vh, T1);

    // Step 3: sample a point on the projected unit disk (Heitz §4.2).
    let r   = sqrt(xi.x);
    let phi = TWO_PI * xi.y;
    let t1  = r * cos(phi);
    var t2  = r * sin(phi);

    // s blends cosine weighting (s=0, view grazes surface) against uniform disk
    // sampling (s=1, view along normal), tracking the projected visible area.
    let s = 0.5 * (1.0 + Vh.z);
    t2 = mix(sqrt(max(0.0, 1.0 - t1 * t1)), t2, s);

    // Step 4: reproject sample onto the hemisphere, then unstretch.
    let Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    return normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
}

// ============================================================================
// BRDF helpers
// ============================================================================

// GGX (Trowbridge-Reitz) normal distribution function. `alpha` = roughness².
// Returns the statistical probability that a given half-vector H aligns with
// a microfacet normal — the "D" term in the Cook-Torrance BRDF.
fn d_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2    = alpha * alpha;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Smith G1 shadowing term for GGX. Gives the probability that a direction v
// (with surface normal angle cos_v) is not occluded by a microfacet.
fn smith_g1(cos_v: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    return 2.0 * cos_v / (cos_v + sqrt(a2 + (1.0 - a2) * cos_v * cos_v));
}

// Height-correlated Smith G2 (Heitz 2014). Better than G1²: accounts for the
// correlation between masking and shadowing at the same height. This is the
// "G" term in Cook-Torrance.
fn smith_g2(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let g_v = n_dot_l * sqrt(a2 + (1.0 - a2) * n_dot_v * n_dot_v);
    let g_l = n_dot_v * sqrt(a2 + (1.0 - a2) * n_dot_l * n_dot_l);
    return 2.0 * n_dot_v * n_dot_l / (g_v + g_l);
}

// Schlick approximation to the Fresnel equations. F0 is the surface
// reflectance at normal incidence (computed from IOR for dielectrics,
// taken from albedo for metals).
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let x = clamp(1.0 - cos_theta, 0.0, 1.0);
    let x2 = x * x;
    return f0 + (vec3(1.0) - f0) * (x2 * x2 * x);  // (1-cos)^5
}

// Exact Fresnel equations for a dielectric (glass-like) interface.
// cos_i: cosine of the angle of incidence (must be >= 0).
// eta:   ratio n1/n2 (e.g. 1.0/1.5 when entering glass from air).
// Returns the fraction of light that is REFLECTED (0 = all transmitted,
// 1 = total internal reflection). The rest is refracted.
fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
    let sin_t2 = eta * eta * (1.0 - cos_i * cos_i);
    if sin_t2 >= 1.0 {
        return 1.0;  // total internal reflection — nothing gets through
    }
    let cos_t = sqrt(1.0 - sin_t2);
    let rs = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    let rp = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    return 0.5 * (rs * rs + rp * rp);
}

// Perceptual luminance of a linear-light RGB colour (BT.709 primaries).
// Used to convert a colour weight into a scalar probability for path selection.
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// ============================================================================
// Ray–AABB intersection (slab method)
// ============================================================================

fn ray_aabb(
    orig:    vec3<f32>,
    inv_dir: vec3<f32>,
    box_min: vec3<f32>,
    box_max: vec3<f32>,
    t_max:   f32,
) -> bool {
    let t1 = (box_min - orig) * inv_dir;
    let t2 = (box_max - orig) * inv_dir;
    let t_enter = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let t_exit  = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    return t_exit >= max(t_enter, T_MIN) && t_enter < t_max;
}

// ============================================================================
// Ray–triangle intersection (Möller–Trumbore)
// ============================================================================

// Returns vec3(t, u, v) on a hit, vec3(-1, 0, 0) on a miss.
//
// Back-face culling is intentionally disabled (|det| instead of det > eps):
// glass and other transmissive materials need to hit both sides of a mesh,
// so we accept triangles facing either way.
fn ray_triangle(
    orig:  vec3<f32>,
    dir:   vec3<f32>,
    v0:    vec3<f32>,
    v1:    vec3<f32>,
    v2:    vec3<f32>,
    t_max: f32,
) -> vec3<f32> {
    let e1  = v1 - v0;
    let e2  = v2 - v0;
    let h   = cross(dir, e2);
    let det = dot(e1, h);
    if abs(det) < 1e-8 { return vec3(-1.0, 0.0, 0.0); }  // ray parallel to triangle

    let inv_det = 1.0 / det;
    let s = orig - v0;
    let u = dot(s, h) * inv_det;
    if u < 0.0 || u > 1.0 { return vec3(-1.0, 0.0, 0.0); }

    let q = cross(s, e1);
    let v = dot(dir, q) * inv_det;
    if v < 0.0 || u + v > 1.0 { return vec3(-1.0, 0.0, 0.0); }

    let t = dot(e2, q) * inv_det;
    if t < T_MIN || t >= t_max { return vec3(-1.0, 0.0, 0.0); }

    return vec3(t, u, v);
}

// ============================================================================
// BVH traversal
// ============================================================================

struct HitRecord {
    t:       f32,
    tri_idx: u32,
    u:       f32,
    v:       f32,
    hit:     u32,
}

// Iterative depth-first traversal. 64-deep stack handles any reasonably
// balanced BVH; pathological inputs might overflow but real scenes won't.
fn traverse_bvh(
    orig:    vec3<f32>,
    dir:     vec3<f32>,
    inv_dir: vec3<f32>,
) -> HitRecord {
    var result = HitRecord(T_MAX, 0u, 0.0, 0.0, 0u);

    if arrayLength(&bvh_nodes) == 0u { return result; }

    var stack: array<u32, 64>;
    var sp: i32 = 0;
    stack[sp] = 0u;
    sp += 1;

    while sp > 0 {
        sp -= 1;
        let idx  = stack[sp];
        let node = bvh_nodes[idx];

        if !ray_aabb(orig, inv_dir, node.aabb_min, node.aabb_max, result.t) {
            continue;
        }

        if node.count > 0u {
            for (var i = 0u; i < node.count; i++) {
                let ti  = node.right_or_first + i;
                let tri = bvh_tris[ti];
                let v0  = vec3(tri.v0x, tri.v0y, tri.v0z);
                let v1  = vec3(tri.v1x, tri.v1y, tri.v1z);
                let v2  = vec3(tri.v2x, tri.v2y, tri.v2z);
                let h   = ray_triangle(orig, dir, v0, v1, v2, result.t);
                if h.x >= 0.0 {
                    result = HitRecord(h.x, ti, h.y, h.z, 1u);
                }
            }
        } else {
            stack[sp] = node.right_or_first;
            sp += 1;
            stack[sp] = idx + 1u;
            sp += 1;
        }
    }

    return result;
}

// ============================================================================
// Sky and environment
// ============================================================================

// ============================================================================
// Sky / environment
// ============================================================================

// Samples the HDR environment map using an equirectangular (lat-long) mapping,
// with bilinear filtering.
//
// Standard convention used by Poly Haven and most HDRI tools:
//   longitude = atan2(dir.x, -dir.z)  →  U = 0.5 at -Z (camera forward)
//   latitude  = asin(dir.y)           →  V = 0 at top (+Y), 1 at bottom (-Y)
//
// Bilinear filtering is crucial for smooth reflections in metallic surfaces.
// Without it, adjacent surface normals with tiny angular differences can sample
// completely different env map pixels, creating a stippled grain that persists
// even at high sample counts.
fn sample_env(dir: vec3<f32>) -> vec3<f32> {
    let phi   = atan2(dir.x, -dir.z);                   // [-π, π]
    let theta = asin(clamp(dir.y, -0.9999, 0.9999));    // [-π/2, π/2]

    let u = phi   / TWO_PI + 0.5;           // [0, 1]
    let v = 0.5   - theta  / PI;            // [0, 1], 0 = top

    let W = frame.env_width;
    let H = frame.env_height;

    // Map to texel space, offset -0.5 for texel-centre alignment.
    let uf = u * f32(W) - 0.5;
    let vf = v * f32(H) - 0.5;

    let x0i = i32(floor(uf));
    let y0i = i32(floor(vf));

    // Wrap U (equirectangular is periodic), clamp V (poles).
    let x0 = u32(((x0i     % i32(W)) + i32(W)) % i32(W));
    let x1 = u32((((x0i + 1) % i32(W)) + i32(W)) % i32(W));
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

// Returns the radiance for a ray that escaped the scene without hitting
// any geometry.
//
// When an HDR environment map is loaded (env_width > 0) AND IBL is not
// disabled (flags bit 1 == 0), the map is sampled directly — giving
// physically-based ambient lighting, correct coloured reflections in
// metals, and a background that matches the IBL.
//
// IBL is toggled by bit 1 of frame.flags so the user can quickly compare
// the env-lit render against the procedural sky without reloading.
//
// The sun is handled exclusively by NEE (see main path loop) so it does
// not appear here regardless of which sky path is taken.
fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let ibl_disabled = (frame.flags & 2u) != 0u;
    if frame.env_width > 0u && !ibl_disabled {
        return sample_env(dir);
    }
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(SKY_HORIZON, SKY_ZENITH, t);
}

// ============================================================================
// Multiple Importance Sampling (MIS) helpers
// ============================================================================

// Power heuristic (Veach 1997, β=2). Blends two sampling strategies with PDF
// `pdf_a` and `pdf_b` so that the combined estimator minimises variance.
//
// w(a, b) = a² / (a² + b²)
//
// β=2 is more aggressive than the balance heuristic (β=1): it down-weights the
// lower-PDF strategy harder, which reduces variance when the two PDFs are very
// different in magnitude. For IBL + BRDF this is almost always the right choice.
fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2);
}

// Importance-sample the environment map using the pre-built 2-D CDF.
//
// Returns vec4(direction.xyz, pdf_solid_angle). The direction is the world-space
// direction of the sampled env pixel; pdf is in solid-angle measure so it can be
// compared directly with BRDF PDFs in the power heuristic.
//
// Algorithm: two-stage inversion method.
//   1. Binary-search the marginal CDF (over rows) for a row index.
//   2. Binary-search the conditional CDF (over columns within that row).
//   3. Convert the pixel centre (col, row) back to a direction via the inverse
//      of the equirectangular latitude-longitude mapping used in sample_env().
fn sample_env_importance(xi: vec2<f32>) -> vec4<f32> {
    let W = frame.env_width;
    let H = frame.env_height;

    // ── Stage 1: find row via marginal CDF binary search ─────────────────────
    // Lower-bound search: first index where cdf[mid] >= xi.x.
    var row_lo = 0u; var row_hi = H;
    while row_lo < row_hi {
        let mid = (row_lo + row_hi) >> 1u;
        if env_marginal_cdf[mid] < xi.x { row_lo = mid + 1u; } else { row_hi = mid; }
    }
    let row = min(row_lo, H - 1u);

    // ── Stage 2: find column via conditional CDF binary search ────────────────
    let row_off = row * W;
    var col_lo = 0u; var col_hi = W;
    while col_lo < col_hi {
        let mid = (col_lo + col_hi) >> 1u;
        if env_conditional_cdf[row_off + mid] < xi.y { col_lo = mid + 1u; } else { col_hi = mid; }
    }
    let col = min(col_lo, W - 1u);

    // ── Convert pixel centre to world-space direction ─────────────────────────
    // Inverse of sample_env()'s direction → UV mapping:
    //   phi   = atan2(dir.x, -dir.z)  →  u = phi/(2π) + 0.5
    //   theta = asin(dir.y)           →  v = 0.5 - theta/π
    // Rearranging: phi = (u-0.5)·2π, theta = (0.5-v)·π (latitude, not polar).
    //   dir.y = sin(theta)
    //   dir.x = cos(theta)·sin(phi)    from phi = atan2(dir.x, -dir.z)
    //   dir.z = -cos(theta)·cos(phi)
    let u = (f32(col) + 0.5) / f32(W);
    let v = (f32(row) + 0.5) / f32(H);
    let phi     = (u - 0.5) * TWO_PI;
    let lat     = (0.5 - v) * PI;         // latitude = asin(dir.y)
    let cos_lat = cos(lat);
    let dir     = normalize(vec3(cos_lat * sin(phi), sin(lat), -cos_lat * cos(phi)));

    // ── PDF in solid-angle measure ────────────────────────────────────────────
    // pdf_ω = luminance(pixel) * env_sample_weight.
    // (The sin(θ) terms cancel between pdf_pixel and dω — see build_env_cdfs.)
    let pix_lum = luminance(env_pixels[row * W + col].rgb);
    let pdf     = max(pix_lum * frame.env_sample_weight, 0.0);

    return vec4(dir, pdf);
}

// Evaluate the PDF of sampling direction `dir` under the env-map distribution.
//
// Used in the miss case to compute the MIS weight for BRDF-sampled escape rays:
// once we know the ray escaped to the env, we ask "what PDF would the env
// importance sampler have assigned to this direction?" and blend accordingly.
fn eval_env_pdf(dir: vec3<f32>) -> f32 {
    if frame.env_sample_weight <= 0.0 { return 0.0; }
    return luminance(sample_env(dir)) * frame.env_sample_weight;
}

// Evaluate the combined diffuse+specular BRDF × cos(θ_i) for direction `l`
// given viewer `v`, surface normal `n`, linearised roughness `alpha`,
// Fresnel F0 `f0`, base colour `albedo`, and metallic factor `metallic`.
//
// Returns `(f_diffuse + f_specular) * dot(n, l)` — the integrand factor for
// the rendering equation (BRDF times the projected solid angle cosine term).
// This is what gets multiplied by env_radiance and divided by pdf_env when
// evaluating the explicit NEE contribution.
fn eval_brdf_f(
    l:       vec3<f32>,  // incident direction (toward light)
    n:       vec3<f32>,  // shading normal
    v:       vec3<f32>,  // view direction (toward camera)
    alpha:   f32,        // linearised roughness = roughness²
    f0:      vec3<f32>,  // Fresnel reflectance at normal incidence
    albedo:  vec3<f32>,
    metallic: f32,
) -> vec3<f32> {
    let n_dot_l = max(dot(n, l), 0.0);
    if n_dot_l <= 0.0 { return vec3(0.0); }

    let h       = normalize(v + l);
    let n_dot_v = max(dot(n, v), 0.0001);
    let n_dot_h = max(dot(n, h), 0.0001);
    let v_dot_h = max(dot(v, h), 0.0001);

    let F        = fresnel_schlick(v_dot_h, f0);
    let D        = d_ggx(n_dot_h, alpha);
    let G        = smith_g2(n_dot_v, n_dot_l, alpha);
    let specular = F * D * G / max(4.0 * n_dot_v * n_dot_l, 0.0001);
    let diffuse  = (1.0 - metallic) * albedo * (vec3(1.0) - F) * INV_PI;

    return (specular + diffuse) * n_dot_l;
}

// Evaluate the mixed BRDF PDF for direction `l`, combining the GGX VNDF specular
// lobe (sampled with probability `p_spec`) and the cosine-weighted diffuse
// hemisphere (sampled with probability `1 - p_spec`).
//
// This is the "partner PDF" used when env-map importance sampling selects a
// direction: we need to know how likely the VNDF path would have been to
// send a ray in that direction in order to compute the power heuristic weight.
fn eval_brdf_pdf(
    l:       vec3<f32>,  // candidate direction (toward light)
    n:       vec3<f32>,  // shading normal
    v:       vec3<f32>,  // view direction
    alpha:   f32,        // linearised roughness
    n_dot_v: f32,        // dot(n, v), precomputed by caller
    p_spec:  f32,        // probability of taking the specular branch
) -> f32 {
    let n_dot_l = max(dot(n, l), 0.0);
    let h       = normalize(v + l);
    let n_dot_h = max(dot(n, h), 0.0001);

    // VNDF specular PDF: D(H) · G1(N·V) / (4 · N·V).
    // Derived from VNDF(H|V) = D(H)·G1(V·H)·(V·H)/(N·V) plus the Jacobian
    // dH/dL = 1/(4·V·H); the V·H terms cancel, giving a clean expression.
    // (The old NDF formula was D(H)·(N·H)/(4·V·H) — this is its VNDF replacement.)
    let pdf_spec = d_ggx(n_dot_h, alpha) * smith_g1(n_dot_v, alpha) / max(4.0 * n_dot_v, 0.0001);
    // Lambertian diffuse PDF: cos(θ_l) / π.
    let pdf_diff = n_dot_l * INV_PI;

    return p_spec * pdf_spec + (1.0 - p_spec) * pdf_diff;
}

// ============================================================================
// Normal mapping
// ============================================================================

// Apply a normal map using the stored per-vertex tangent (GLTF convention).
//
// GLTF normal maps are baked against Mikktspace tangents, which are stored
// in the file alongside normals. Using the stored tangent — rather than
// deriving one analytically from triangle edges — gives correct results at UV
// seams and on mirrored UV islands, where the cotangent method can silently
// produce the wrong tangent direction.
//
// T4.w is the handedness bit (+1 or -1). It flips the bitangent for UV islands
// whose winding is mirrored, which is exactly the case the analytical method
// gets wrong.
//
// Falls back to the unperturbed smooth normal if the texture index is negative
// (no map), the map sample is degenerate, or the tangent is a zero vector
// (vertex had no tangent data in the source file).
fn apply_normal_map(
    tex_idx:       i32,
    uv:            vec2<f32>,
    smooth_normal: vec3<f32>,
    T4:            vec4<f32>,  // xyz = world-space tangent, w = handedness
) -> vec3<f32> {
    if tex_idx < 0 { return smooth_normal; }

    // Sample the normal map. Values are in [0,1]; remap to tangent-space [-1,1].
    let ts = sample_texture(tex_idx, uv).xyz * 2.0 - 1.0;
    if dot(ts, ts) < 0.01 { return smooth_normal; }  // blank/degenerate map

    // Guard against meshes that shipped without tangent data.
    if dot(T4.xyz, T4.xyz) < 0.01 { return smooth_normal; }

    let T = normalize(T4.xyz);
    // Bitangent = cross(N, T) * handedness. The sign flip corrects mirrored UVs.
    //
    // Guard: if the stored tangent happens to be parallel to the normal (which
    // shouldn't happen with well-formed GLTF tangents, but can with degenerate
    // or hand-edited meshes), the cross product is zero and normalize(zero) = NaN.
    let B_raw = cross(smooth_normal, T);
    if dot(B_raw, B_raw) < 1e-6 { return smooth_normal; }
    let B = normalize(B_raw) * T4.w;

    // TBN rotates tangent-space → world-space.
    return normalize(T * ts.x + B * ts.y + smooth_normal * ts.z);
}

// ============================================================================
// Tone mapping and output packing
// ============================================================================

// ACES filmic tone mapping (Narkowicz 2015 fit). Maps HDR linear-light values
// into [0,1] in a way that preserves contrast and has a natural film-like
// S-curve. Much better than the sqrt gamma hack it replaces.
fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3(0.0), vec3(1.0));
}

// Pack a linear-light HDR colour into a RGBA8 u32 for canvas output.
// Applies exposure compensation, ACES tone mapping, then gamma 2.2 encoding.
//
// Exposure is applied as linear *= 2^stops before tone-mapping, so +1 EV
// doubles the brightness and −1 EV halves it. Because this runs on the
// per-sample average (not on the raw accumulation buffer), changing exposure
// mid-render doesn't invalidate accumulated samples — the next dispatch will
// show the updated brightness without restarting.
fn pack_rgba8(linear: vec3<f32>) -> u32 {
    let exposed = linear * pow(2.0, frame.exposure);
    let mapped  = aces(exposed);
    // sRGB gamma: a proper 2.2 power-law (close enough to the piecewise sRGB
    // curve; the difference is invisible at 8 bits per channel).
    let enc = pow(mapped, vec3(1.0 / 2.2));
    let r = u32(enc.r * 255.0 + 0.5);
    let g = u32(enc.g * 255.0 + 0.5);
    let b = u32(enc.b * 255.0 + 0.5);
    return r | (g << 8u) | (b << 16u) | (255u << 24u);
}

// ============================================================================
// Entry point
// ============================================================================

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = frame.width;
    let h = frame.height;
    if gid.x >= w || gid.y >= h { return; }

    let px  = gid.x;
    let py  = gid.y;
    let idx = py * w + px;

    // ── Per-sample RNG seed ───────────────────────────────────────────────────
    var seed = (px * 1973u) ^ (py * 9277u) ^ (frame.sample_index * 26699u + 1u);

    // ── Jittered primary ray ──────────────────────────────────────────────────
    let jx = rand_f32(&seed) - 0.5;
    let jy = rand_f32(&seed) - 0.5;

    let u = (f32(px) + 0.5 + jx) / f32(w);
    let v = (f32(py) + 0.5 + jy) / f32(h);

    // Pinhole ray direction from pixel UV.
    let pinhole_dir = normalize(
        camera.fwd
        + (2.0 * u - 1.0) * camera.rgt_scaled
        + (1.0 - 2.0 * v) * camera.up_scaled,
    );

    // ── Thin-lens depth of field ──────────────────────────────────────────────
    // When aperture == 0 (default) this is a pinhole camera — every point in
    // the scene is in focus. With aperture > 0 we model a thin lens:
    //   1. The pinhole ray defines the focus point on the focal plane.
    //   2. The ray origin is jittered uniformly across a disc of radius
    //      `aperture` in the camera's image plane.
    //   3. The new direction converges at the same focus point, so geometry
    //      exactly on the focal plane stays sharp while everything else blurs
    //      into a circle of confusion proportional to its depth offset.
    //
    // `rgt_scaled` and `up_scaled` are scaled FOV basis vectors; normalising
    // them gives unit right/up vectors in world space for the lens disc.
    var ray_orig = camera.position;
    var ray_dir  = pinhole_dir;
    if frame.aperture > 0.0 {
        let focus_pt   = camera.position + pinhole_dir * frame.focus_dist;
        let lens_r     = frame.aperture * sqrt(rand_f32(&seed));
        let lens_theta = TWO_PI * rand_f32(&seed);
        let cam_right  = normalize(camera.rgt_scaled);
        let cam_up     = normalize(camera.up_scaled);
        ray_orig = camera.position
            + lens_r * (cos(lens_theta) * cam_right + sin(lens_theta) * cam_up);
        ray_dir  = normalize(focus_pt - ray_orig);
    }

    // ── Path tracing loop ─────────────────────────────────────────────────────
    //
    // `throughput` is the accumulated transmittance of the path so far —
    // the fraction of light at the current bounce that will reach the camera
    // if nothing else happens. It starts at white (full transmission) and gets
    // multiplied by the BRDF / PDF weight at each bounce.
    //
    // `radiance` accumulates the light contributions at each bounce. When the
    // path hits an emissive surface or the sky, the emission is weighted by
    // the current throughput and added to the running total.
    //
    // In preview mode (frame.flags bit 0) we cap the bounce count to 2 —
    // primary ray + one indirect bounce. This is ~4-6× cheaper than the full
    // bounce count and still shows albedo, reflections, and basic lighting.
    // Otherwise we use frame.max_bounces, set by the user's slider (default 12).
    let preview_mode = (frame.flags & 1u) != 0u;
    let max_bounces  = select(i32(frame.max_bounces), PREVIEW_BOUNCES, preview_mode);

    // MIS is active when an env map is loaded, IBL is enabled, AND we are not
    // in preview mode (preview prioritises speed over accuracy).
    let ibl_disabled = (frame.flags & 2u) != 0u;
    let do_mis = !preview_mode && frame.env_width > 0u && !ibl_disabled
              && frame.env_sample_weight > 0.0;

    // Compute the sun direction from the user-controlled azimuth and elevation
    // uniforms. Standard spherical-to-Cartesian with Y-up convention:
    //   azimuth  = horizontal angle from +Z toward +X (like a compass bearing)
    //   elevation = angle above the horizontal plane
    // normalize() handles floating-point drift near the horizon or zenith.
    let cos_el  = cos(frame.sun_elevation);
    let sun_dir = normalize(vec3(
        sin(frame.sun_azimuth)  * cos_el,   // X component
        sin(frame.sun_elevation),            // Y component (height)
        cos(frame.sun_azimuth)  * cos_el,   // Z component
    ));

    var throughput = vec3(1.0);
    var radiance   = vec3(0.0);

    // BRDF PDF of the ray that arrived at the current bounce.
    //
    // Set to -1.0 when there is no valid BRDF PDF for the arriving ray:
    //   • Before the first hit (primary camera ray — no previous scatter).
    //   • After a glass bounce (Dirac-delta transmissive/reflective BRDF).
    // When >= 0, the value is used in the miss case to weight the implicit
    // env-map contribution against the explicit env NEE from the previous hit.
    var mis_brdf_pdf: f32 = -1.0;

    for (var bounce: i32 = 0; bounce < max_bounces; bounce++) {
        let inv_dir = vec3(1.0 / ray_dir.x, 1.0 / ray_dir.y, 1.0 / ray_dir.z);
        let hit = traverse_bvh(ray_orig, ray_dir, inv_dir);

        if hit.hit == 0u {
            // ── Ray escaped the scene ─────────────────────────────────────────
            //
            // With MIS, the env map is sampled explicitly (NEE) at each surface
            // hit, so escaped rays get a MIS weight that prevents double-counting
            // the same env light through both strategies.
            //
            // When env_background == 0, the user wants IBL lighting without a
            // visible background. The env NEE at surface hits still runs (giving
            // indirect IBL), but the escaped ray contributes dark grey — so the
            // background stays dark and only geometry lit by the env map is lit.
            if frame.env_background == 1u {
                let env_rad = sky_color(ray_dir);
                if do_mis && mis_brdf_pdf >= 0.0 {
                    // Weight the implicit BRDF-sampled env contribution against the
                    // explicit env NEE that fired at the previous surface hit.
                    // If the env PDF is near zero (black direction), give full weight
                    // to the implicit path to avoid losing dark-region IBL.
                    let pdf_env = eval_env_pdf(ray_dir);
                    let w = select(1.0, power_heuristic(mis_brdf_pdf, pdf_env), pdf_env > 1e-8);
                    radiance += throughput * env_rad * w * frame.ibl_scale;
                } else {
                    radiance += throughput * env_rad * frame.ibl_scale;
                }
            } else {
                // Dark studio grey: neutral but not pitch-black so silhouettes
                // remain legible against the background.
                radiance += throughput * vec3<f32>(0.04, 0.04, 0.04);
            }
            break;
        }

        let tri = bvh_tris[hit.tri_idx];
        let mat = materials[tri.material_index];

        // ── Hit point geometry ────────────────────────────────────────────────

        let hit_pos = ray_orig + ray_dir * hit.t;

        // Barycentric interpolation of vertex attributes.
        let bary_w = 1.0 - hit.u - hit.v;

        let n0 = vec3(tri.n0x, tri.n0y, tri.n0z);
        let n1 = vec3(tri.n1x, tri.n1y, tri.n1z);
        let n2 = vec3(tri.n2x, tri.n2y, tri.n2z);
        let smooth_n = normalize(bary_w * n0 + hit.u * n1 + hit.v * n2);

        let uv = bary_w * vec2(tri.uv0u, tri.uv0v)
               + hit.u  * vec2(tri.uv1u, tri.uv1v)
               + hit.v  * vec2(tri.uv2u, tri.uv2v);

        // ── Material properties ───────────────────────────────────────────────

        // GLTF spec: albedo and emissive textures are sRGB-encoded (gamma ~2.2).
        // Decode to linear light before any math. Metallic/roughness and normal
        // maps are linear data and must NOT be decoded.
        let albedo_tex_samp = pow(sample_texture(mat.albedo_tex, uv).rgb, vec3(2.2));
        var albedo    = mat.albedo.rgb * albedo_tex_samp;
        let emissive  = vec3(mat.emissive_r, mat.emissive_g, mat.emissive_b)
                      * pow(sample_texture(mat.emissive_tex, uv).rgb, vec3(2.2));

        // Metallic/roughness texture: blue = metallic, green = roughness (GLTF spec).
        let mr_sample = sample_texture(mat.mr_tex, uv);
        let metallic  = mat.metallic  * mr_sample.b;
        let roughness = max(mat.roughness * mr_sample.g, 0.04); // clamp to avoid singularities
        let alpha     = roughness * roughness;  // perceptual → linear roughness

        // Normal map — barycentrically interpolate the stored per-vertex tangents,
        // then decode the map in the correct Mikktspace tangent basis.
        //
        // Handedness (w) is constant across a triangle — it encodes whether
        // the UV island is mirrored, which is a per-island property, not per-
        // vertex. Taking t0.w is correct; we just need the xyz interpolated.
        let t0 = vec4(tri.t0x, tri.t0y, tri.t0z, tri.t0w);
        let t1 = vec4(tri.t1x, tri.t1y, tri.t1z, tri.t1w);
        let t2 = vec4(tri.t2x, tri.t2y, tri.t2z, tri.t2w);
        // Interpolate tangent directions barycentrically.
        //
        // At UV seams, adjacent triangles belong to different UV islands and
        // their per-vertex tangents can point in opposite directions. Naive
        // barycentric interpolation across such a boundary produces a near-zero
        // sum, and normalize(zero) = NaN. NaN in the accumulation buffer is
        // permanent (NaN + x = NaN forever), so it must be stopped here.
        //
        // The fix: skip normalize when the interpolated tangent is too short.
        // The zero tangent falls through to apply_normal_map's own "no tangent
        // data" guard (dot < 0.01), which returns smooth_normal unchanged.
        let tangent_raw = bary_w * t0.xyz + hit.u * t1.xyz + hit.v * t2.xyz;
        let tangent = vec4(
            select(vec3(0.0), normalize(tangent_raw), dot(tangent_raw, tangent_raw) > 1e-6),
            t0.w,
        );
        let shading_n = apply_normal_map(mat.normal_tex, uv, smooth_n, tangent);

        // ── Emission ──────────────────────────────────────────────────────────
        radiance += throughput * emissive;

        // ── Russian roulette ──────────────────────────────────────────────────
        //
        // After RR_START bounces, randomly terminate the path with probability
        // proportional to how dim the path has become. Surviving paths get their
        // throughput boosted to keep the estimator unbiased. This avoids wasting
        // time tracing paths that contribute almost nothing, while preserving the
        // expected value of the estimator.
        if bounce >= RR_START {
            let p = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.01, 1.0);
            if rand_f32(&seed) > p { break; }
            throughput /= p;
        }

        // ── Scatter event selection ───────────────────────────────────────────
        //
        // We stochastically choose among three event types:
        //   1. Transmission (glass): ray passes through the surface.
        //   2. Specular reflection: ray reflects off the microsurface (GGX BRDF).
        //   3. Diffuse scattering: ray scatters into a cosine-weighted hemisphere.
        //
        // The probabilities are derived from the material properties so the
        // estimator remains unbiased (each event is weighted by 1/probability).

        if mat.transmission > 0.001 && rand_f32(&seed) < mat.transmission {
            // ── Glass / transmission ──────────────────────────────────────────
            //
            // Determine whether the ray is entering or exiting the glass by
            // checking if the surface normal faces toward or away from the ray.
            let entering  = dot(ray_dir, shading_n) < 0.0;
            let geo_n     = select(-shading_n, shading_n, entering);
            let eta       = select(mat.ior, 1.0 / mat.ior, entering);
            let cos_i     = abs(dot(-ray_dir, geo_n));

            // Exact dielectric Fresnel: decides reflect vs refract.
            let F = fresnel_dielectric(cos_i, eta);

            var new_dir: vec3<f32>;
            if rand_f32(&seed) < F {
                // Fresnel reflection.
                new_dir = reflect(ray_dir, geo_n);
            } else {
                // Snell's law refraction. Check for total internal reflection
                // (shouldn't happen since F handled it, but guard anyway).
                let sin_t2 = eta * eta * (1.0 - cos_i * cos_i);
                if sin_t2 >= 1.0 {
                    new_dir = reflect(ray_dir, geo_n);
                } else {
                    let cos_t = sqrt(1.0 - sin_t2);
                    new_dir   = normalize(eta * ray_dir + (eta * cos_i - cos_t) * geo_n);
                    // Glass tints transmitted light.
                    throughput *= albedo;
                }
            }

            // Glass uses a Dirac-delta BRDF — the transmitted/reflected direction
            // is determined exactly, not from a PDF. MIS requires a finite PDF to
            // compare against the env PDF, so disable it for the next bounce.
            mis_brdf_pdf = -1.0;

            // Offset origin slightly in the transmission direction so the next
            // bounce starts on the correct side of the surface.
            ray_orig = hit_pos + new_dir * (T_MIN * 10.0);
            ray_dir  = new_dir;

        } else {
            // ── Opaque PBR surface ────────────────────────────────────────────
            //
            // The surface must be on the same side as the incoming ray. If the
            // smooth normal points away (can happen at silhouette edges), flip it.
            let n = select(-shading_n, shading_n, dot(-ray_dir, shading_n) > 0.0);

            let v_dir    = -ray_dir;                    // vector from surface to viewer
            let n_dot_v  = max(dot(n, v_dir), 0.0001); // avoid division by zero

            // Fresnel at normal incidence: 0.04 for dielectrics, albedo for metals.
            let f0 = mix(vec3(0.04), albedo, metallic);
            let F_approx = fresnel_schlick(n_dot_v, f0);

            // Probability of taking the specular path.
            //
            // Using luminance(F) alone breaks for dark metallic surfaces: a dark
            // olive-green metal (metallic=1, albedo=[0.02,0.05,0.01]) has
            // luminance(f0) ≈ 0.04, so p_spec clamps to 0.04. That means 96 % of
            // paths take the diffuse branch, multiply throughput by
            // (1–metallic)*albedo*(1–F) = 0, and die — contributing nothing but
            // consuming a sample slot. The surviving 4 % get a 25× Fresnel boost
            // at grazing angles, producing explosive variance that never converges.
            //
            // Fix: take max(metallic, luminance(F)) so fully metallic surfaces always
            // sample the specular lobe. The diffuse branch can still fire (4 % of the
            // time) and correctly returns zero throughput, but variance is contained.
            let p_spec = clamp(max(metallic, luminance(F_approx)), 0.04, 0.96);

            // ── Direct sun lighting (Next Event Estimation) ───────────────────
            //
            // Fire one shadow ray toward the sun at every opaque surface hit.
            // If nothing blocks it, evaluate the full PBR BRDF at the sun angle
            // and add the contribution directly — no lottery required.
            //
            // Without NEE the sun is a disk covering ~1/800th of the hemisphere.
            // A diffuse bounce has a 1-in-800 chance of stumbling onto it, so
            // most pixels stay dark and a few are explosively bright — classic
            // Monte Carlo variance. NEE trades that for one extra BVH traversal
            // per bounce and a completely smooth direct-lighting result.
            //
            // The sun is a Dirac delta (zero area from our perspective), so no
            // MIS weight is applied — the probability of a random BRDF scatter
            // sampling the sun exactly is effectively zero regardless of the BRDF.
            //
            // Skip the analytical sun when an HDR environment map is active (do_mis).
            // HDR maps already contain a physical sun (or equivalent) and the env
            // importance sampler handles it correctly. Running both simultaneously
            // double-counts the sun: metallic specular highlights get 2–3× too much
            // energy, spiking past the firefly clamp and producing persistent grain.
            let n_dot_sun = dot(n, sun_dir);
            if n_dot_sun > 0.0 && !do_mis {
                let shadow_orig = hit_pos + n * T_MIN;
                let shadow_inv  = vec3(1.0 / sun_dir.x, 1.0 / sun_dir.y, 1.0 / sun_dir.z);
                let shadow_hit  = traverse_bvh(shadow_orig, sun_dir, shadow_inv);

                if shadow_hit.hit == 0u {
                    // Sun is unoccluded — evaluate diffuse + specular at the sun angle.
                    let H_sun   = normalize(v_dir + sun_dir);
                    let v_dot_h = max(dot(v_dir, H_sun), 0.0001);
                    let n_dot_h = max(dot(n, H_sun), 0.0001);

                    // Fresnel at the view/half-vector angle.
                    let F_sun = fresnel_schlick(v_dot_h, f0);

                    // Lambertian diffuse: albedo/π, zero for metals, weighted by
                    // the fraction of energy not claimed by Fresnel reflection.
                    let diffuse_sun = albedo * (1.0 - metallic) * (vec3(1.0) - F_sun) * INV_PI;

                    // Cook-Torrance specular: D * F * G / (4 · NdotV · NdotL).
                    let D_sun = d_ggx(n_dot_h, alpha);
                    let G_sun = smith_g2(n_dot_v, n_dot_sun, alpha);
                    let specular_sun = F_sun * D_sun * G_sun
                                     / max(4.0 * n_dot_v * n_dot_sun, 0.0001);

                    radiance += throughput * (diffuse_sun + specular_sun)
                              * n_dot_sun * SUN_RADIANCE * frame.sun_intensity;
                }
            }

            // ── Environment NEE (MIS explicit sampling) ───────────────────────
            //
            // When an HDR env map is loaded and IBL is enabled, fire a shadow ray
            // toward an importance-sampled env direction. This is the "explicit"
            // half of the two-strategy MIS estimator — the "implicit" half is the
            // BRDF-sampled ray that reaches the env in the miss case.
            //
            // Power heuristic weight:
            //   w_env = pdf_env² / (pdf_env² + pdf_brdf²)
            //
            // For a smooth metallic surface pointing directly at a bright env
            // pixel, pdf_brdf is high (BRDF sampler would likely pick that
            // direction anyway) and pdf_env is moderate, so w_env is lower —
            // the explicit NEE contributes less because the BRDF sampler already
            // handles it well.  For rough diffuse surfaces, pdf_brdf is broadly
            // spread but pdf_env is concentrated on bright pixels, so w_env ≈ 1
            // and NEE gives all the bright contribution — exactly what's needed
            // to avoid fireflies.
            if do_mis {
                let env_sample = sample_env_importance(rand2(&seed));
                let env_dir    = env_sample.xyz;
                let pdf_env    = env_sample.w;
                let n_dot_env  = dot(n, env_dir);

                if n_dot_env > 0.0 && pdf_env > 1e-8 {
                    let env_shadow_orig = hit_pos + n * T_MIN;
                    let env_inv_dir     = vec3(1.0 / env_dir.x, 1.0 / env_dir.y, 1.0 / env_dir.z);
                    let env_shadow_hit  = traverse_bvh(env_shadow_orig, env_dir, env_inv_dir);

                    if env_shadow_hit.hit == 0u {
                        let env_radiance = sample_env(env_dir);
                        let pdf_brdf     = eval_brdf_pdf(env_dir, n, v_dir, alpha, n_dot_v, p_spec);
                        let w_env        = power_heuristic(pdf_env, pdf_brdf);
                        let brdf_f       = eval_brdf_f(env_dir, n, v_dir, alpha, f0, albedo, metallic);
                        radiance += throughput * brdf_f * env_radiance / pdf_env * w_env * frame.ibl_scale;
                    }
                }
            }

            var new_dir: vec3<f32>;

            if rand_f32(&seed) < p_spec {
                // ── GGX specular (VNDF sampled) ───────────────────────────────
                //
                // VNDF-sample a microfacet half-vector H visible from v_dir,
                // then reflect the incoming ray about H to get outgoing direction L.
                // See ggx_vndf() for why this beats plain NDF sampling on smooth metals.
                let onb = make_onb(n);
                // Transform v_dir to local space so VNDF can condition on it.
                let v_local = vec3(dot(v_dir, onb.t), dot(v_dir, onb.b), dot(v_dir, n));
                let H_local = ggx_vndf(v_local, alpha, rand2(&seed));
                let H = onb_to_world(onb, n, H_local);

                new_dir = reflect(-v_dir, H);

                // Reject samples where the reflected direction is below the surface.
                let n_dot_l_raw = dot(n, new_dir);
                if n_dot_l_raw <= 0.0 {
                    break;
                }
                let n_dot_l = max(n_dot_l_raw, 0.0001);

                // BRDF weight for VNDF sampling.
                //
                // With NDF sampling the weight was:  F * G2 * (V·H) / (G1 * (N·H))
                // With VNDF sampling the (V·H)/(N·H) factor cancels against the
                // ratio of the VNDF PDF to the NDF PDF, leaving just F * G2 / G1.
                // This is not only simpler but also numerically more stable.
                let v_dot_h = max(dot(v_dir, H), 0.0001);
                let F_spec  = fresnel_schlick(v_dot_h, f0);
                let G2_spec = smith_g2(n_dot_v, n_dot_l, alpha);
                let G1_v    = smith_g1(n_dot_v, alpha);

                throughput *= F_spec * G2_spec / max(G1_v, 1e-6) / p_spec;

                // Store this bounce's BRDF PDF for use in the next miss case.
                mis_brdf_pdf = eval_brdf_pdf(new_dir, n, v_dir, alpha, n_dot_v, p_spec);

            } else {
                // ── Lambertian diffuse ────────────────────────────────────────
                //
                // Metals have no diffuse term (all energy goes into specular).
                // For dielectrics, the diffuse weight is (1-F)*(1-metallic)*albedo.
                let onb = make_onb(n);
                let l_local = cosine_hemisphere(rand2(&seed));
                new_dir = onb_to_world(onb, n, l_local);

                // Cosine hemisphere sampling: BRDF=albedo/pi, PDF=cos/pi → weight=albedo.
                // Additional (1-F)*(1-metallic) factor: energy not taken by Fresnel
                // reflection and not absorbed by the metal's free electrons.
                let F_diff = fresnel_schlick(max(dot(n, new_dir), 0.0), f0);
                throughput *= (1.0 - metallic) * albedo * (vec3(1.0) - F_diff) / (1.0 - p_spec);

                // Store this bounce's BRDF PDF for use in the next miss case.
                mis_brdf_pdf = eval_brdf_pdf(new_dir, n, v_dir, alpha, n_dot_v, p_spec);
            }

            ray_orig = hit_pos + n * T_MIN;
            ray_dir  = new_dir;
        }
    }

    // ── Firefly suppression ───────────────────────────────────────────────────
    //
    // MIS eliminates the main source of env-map fireflies by importance-sampling
    // bright env pixels explicitly at each bounce. The clamp is raised to 10.0
    // (was 3.0) now that MIS handles most of the variance — remaining outliers
    // are typically glass caustics or very high-frequency emissive geometry where
    // MIS doesn't apply. A higher threshold lets the tone-mapper show accurate
    // peak specular highlights on metallic surfaces.
    //
    // At 10.0: ACES(10) ≈ 0.998 (essentially white) — correct for the peak of
    // a mirror reflection. Outliers beyond 10 are genuine sampling accidents
    // that still need suppression to avoid the classic firefly "stuck bright
    // pixel" problem. The bias is now negligible for anything MIS handles.
    //
    // The clamp is applied to the luminance so the hue of bright samples is
    // preserved — we scale all channels equally rather than clamping per-channel.
    let sample_lum = dot(radiance, vec3(0.2126, 0.7152, 0.0722));
    let FIREFLY_CLAMP = 10.0;
    if sample_lum > FIREFLY_CLAMP {
        radiance *= FIREFLY_CLAMP / sample_lum;
    }

    // ── NaN sanitisation ─────────────────────────────────────────────────────
    //
    // NaN in the accumulation buffer is permanent: NaN + anything = NaN, so
    // one bad sample locks the pixel black forever. IEEE 754 guarantees that
    // NaN != NaN is true, so this catches any component that went NaN (from
    // degenerate geometry or numerical edge cases) and treats it as black
    // (contributes nothing to the running average) rather than corrupting the
    // pixel permanently.
    if any(radiance != radiance) { radiance = vec3(0.0); }

    // ── Progressive accumulation ──────────────────────────────────────────────
    let sample = vec4<f32>(radiance, 1.0);
    var total: vec4<f32>;
    if frame.sample_index == 0u {
        total = sample;
    } else {
        total = accum[idx] + sample;
    }
    accum[idx] = total;

    let avg = total.rgb / f32(frame.sample_index + 1u);
    output[idx] = pack_rgba8(avg);
}
