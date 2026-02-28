// Path-tracing compute shader — physically-based, multi-bounce.
//
// One GPU thread per pixel. For each sample it:
//   1. Shoots a jittered primary ray from the camera.
//   2. Finds the closest triangle via BVH traversal.
//   3. At the hit point, evaluates emission, then stochastically chooses a
//      scatter event (diffuse, GGX specular, or glass refraction) weighted
//      by the surface BRDF and the Fresnel equations.
//   4. Spawns the scattered ray and repeats up to MAX_BOUNCES times.
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
// Group 1 — per-frame data (updated every dispatch)
//   binding 0 : Camera uniform     — camera position + pre-scaled basis
//   binding 1 : FrameUniforms      — output width, height, sample index
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
//   offset 48 : emissive_tex, ior, transmission, _pad  (i32 + two f32s + u32)
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
    emissive_tex: i32,
    ior:          f32,
    transmission: f32,
    _pad:         u32,
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

// Matches renderer::gpu::FrameUniforms (16 bytes).
struct FrameUniforms {
    width:        u32,
    height:       u32,
    sample_index: u32,
    flags:        u32,  // bit 0: preview mode (cap bounces to PREVIEW_BOUNCES)
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read>       bvh_nodes: array<BvhNode>;
@group(0) @binding(1) var<storage, read>       bvh_tris:  array<BvhTriangle>;
@group(0) @binding(2) var<storage, read>       materials: array<Material>;
@group(0) @binding(3) var<storage, read>       tex_data:  array<u32>;
@group(0) @binding(4) var<storage, read>       tex_info:  array<TexInfo>;

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

// Maximum bounces in full-quality mode. 12 resolves most glass caustics and
// multi-bounce interiors. More is diminishing returns for typical scenes.
const MAX_BOUNCES: i32 = 12;

// Bounce cap for preview/drag mode. 2 bounces = direct light + one reflection
// or transmission. Cheap enough to stay interactive; still shows PBR materials.
const PREVIEW_BOUNCES: i32 = 2;

// After this bounce index, Russian roulette may terminate the path.
const RR_START: i32 = 3;

// Sun direction (normalised). Points up-and-slightly-left, like a mid-morning
// sun in the northern hemisphere. Baked in as a constant to keep the sky fast.
// normalize(vec3(0.4, 1.0, 0.3)) = vec3(0.3651, 0.9129, 0.1826)... approximately.
const SUN_DIR: vec3<f32> = vec3<f32>(0.3651, 0.9129, 0.1826);

// Sun apparent angular radius in radians. The real sun is ~0.00872 rad (0.5°);
// making it ~2° gives visible caustics without requiring extreme precision.
// cos(0.035) ≈ 0.99939
const SUN_COS: f32 = 0.99939;

// Sky colours at horizon and zenith, in linear light.
const SKY_HORIZON: vec3<f32> = vec3<f32>(0.92, 0.85, 0.75);
const SKY_ZENITH:  vec3<f32> = vec3<f32>(0.20, 0.45, 0.85);

// Sun radiance. Scaled to produce realistic exposure when the sun is visible.
const SUN_RADIANCE: vec3<f32> = vec3<f32>(15.0, 13.5, 10.0);

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

// Sample a texture by index, UV-wrapping, nearest-neighbour.
// tex_idx < 0 means "no texture" — returns white so the caller can multiply
// it against the material's scalar without changing the result.
fn sample_texture(tex_idx: i32, uv: vec2<f32>) -> vec4<f32> {
    if tex_idx < 0 {
        return vec4(1.0, 1.0, 1.0, 1.0);
    }

    let info = tex_info[u32(tex_idx)];
    let uf = fract(uv.x);
    let vf = fract(uv.y);
    let px = min(u32(uf * f32(info.width)),  info.width  - 1u);
    let py = min(u32(vf * f32(info.height)), info.height - 1u);
    let packed = tex_data[info.offset + py * info.width + px];

    return vec4(
        f32( packed        & 0xFFu) / 255.0,
        f32((packed >>  8u) & 0xFFu) / 255.0,
        f32((packed >> 16u) & 0xFFu) / 255.0,
        f32((packed >> 24u) & 0xFFu) / 255.0,
    );
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

// GGX half-vector importance sample. Returns a microfacet normal H in local
// frame. `alpha` is the linearised roughness (roughness²).
// PDF(H) = D_ggx(H) * dot(N,H), where D is the GGX NDF.
fn ggx_sample_h(xi: vec2<f32>, alpha: f32) -> vec3<f32> {
    let phi       = TWO_PI * xi.x;
    // Derivation: invert the CDF of D_ggx (Trowbridge-Reitz distribution)
    let a2        = alpha * alpha;
    let denom     = xi.y * (a2 - 1.0) + 1.0;
    let cos_theta = sqrt((1.0 - xi.y) / max(denom, 1e-7));
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    return vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
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

// Procedural sky: gradient from horizon to zenith, with a sun disk.
// Rays that miss all geometry land here.
fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    // Gradient: 0 = horizon, 1 = zenith.
    let t   = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    var sky = mix(SKY_HORIZON, SKY_ZENITH, t);

    // Sun disk: bright circle + soft glow ring.
    let sun_dot = dot(dir, SUN_DIR);
    let in_disk = select(0.0, 1.0, sun_dot > SUN_COS);
    // A soft halo around the disk (smooth falloff over ~3° ring).
    let halo = smoothstep(SUN_COS - 0.05, SUN_COS, sun_dot) * 0.15;
    sky = sky + (SUN_RADIANCE - sky) * in_disk + halo * SUN_RADIANCE;

    // Prevent the sky below the horizon from contributing negatively.
    return max(sky, vec3(0.0));
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
    let B = normalize(cross(smooth_normal, T)) * T4.w;

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
// Applies ACES tone mapping then gamma 2.2 encoding.
fn pack_rgba8(linear: vec3<f32>) -> u32 {
    let mapped = aces(linear);
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

    var ray_orig = camera.position;
    var ray_dir  = normalize(
        camera.fwd
        + (2.0 * u - 1.0) * camera.rgt_scaled
        + (1.0 - 2.0 * v) * camera.up_scaled,
    );

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
    // primary ray + one indirect bounce. This is ~4-6× cheaper than 12
    // bounces and still shows albedo, reflections, and basic lighting.
    let preview_mode = (frame.flags & 1u) != 0u;
    let max_bounces  = select(MAX_BOUNCES, PREVIEW_BOUNCES, preview_mode);

    var throughput = vec3(1.0);
    var radiance   = vec3(0.0);

    for (var bounce: i32 = 0; bounce < max_bounces; bounce++) {
        let inv_dir = vec3(1.0 / ray_dir.x, 1.0 / ray_dir.y, 1.0 / ray_dir.z);
        let hit = traverse_bvh(ray_orig, ray_dir, inv_dir);

        if hit.hit == 0u {
            // Ray escaped — add sky contribution and terminate.
            radiance += throughput * sky_color(ray_dir);
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

        var albedo    = mat.albedo.rgb * sample_texture(mat.albedo_tex, uv).rgb;
        let emissive  = vec3(mat.emissive_r, mat.emissive_g, mat.emissive_b)
                      * sample_texture(mat.emissive_tex, uv).rgb;

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
        let tangent = vec4(
            normalize(bary_w * t0.xyz + hit.u * t1.xyz + hit.v * t2.xyz),
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

            // Probability of taking the specular path. Clamp away from 0 and 1
            // to avoid zero-probability (division by zero) and 100% one-path scenarios.
            let p_spec = clamp(luminance(F_approx), 0.04, 0.96);

            var new_dir: vec3<f32>;

            if rand_f32(&seed) < p_spec {
                // ── GGX specular ──────────────────────────────────────────────
                //
                // Importance-sample the GGX distribution to get a microfacet
                // half-vector H, then reflect the incoming ray about H to get
                // the outgoing direction L.
                let onb = make_onb(n);
                let H_local = ggx_sample_h(rand2(&seed), alpha);
                let H = onb_to_world(onb, n, H_local);

                new_dir = reflect(-v_dir, H);  // reflect(incident, normal)

                // Reject samples where the reflected direction is below the surface
                // (can happen with rough materials; they just contribute nothing).
                let n_dot_l = dot(n, new_dir);
                if n_dot_l <= 0.0 {
                    break;
                }

                // BRDF weight for GGX importance sampling.
                // With H sampled from D(H)*(N·H), the full expression simplifies to:
                //   weight = F * G2(V,L) * (V·H) / (G1(V) * (N·H))
                // where G2/G1 is the "visibility ratio" that accounts for
                // masking/shadowing of the microsurface.
                let v_dot_h  = max(dot(v_dir, H), 0.0001);
                let n_dot_h  = max(dot(n, H), 0.0001);
                let F_spec   = fresnel_schlick(v_dot_h, f0);
                let G2       = smith_g2(n_dot_v, n_dot_l, alpha);
                let G1_v     = smith_g1(n_dot_v, alpha);
                let vis_weight = G2 * v_dot_h / (G1_v * n_dot_h);

                throughput *= F_spec * vis_weight / p_spec;

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
            }

            ray_orig = hit_pos + n * T_MIN;
            ray_dir  = new_dir;
        }
    }

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
