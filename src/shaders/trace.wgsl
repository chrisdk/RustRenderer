// Path-tracing compute shader — Phase 3: direct lighting, one bounce.
//
// One GPU thread per pixel. Shoots a primary ray, finds the closest triangle
// via iterative BVH traversal, shades with a simple Lambertian BRDF and a
// fixed directional sky light, then writes a packed RGBA8 value to the output
// buffer.
//
// Binding layout
// ──────────────
// Group 0 — scene data (static, uploaded once per loaded scene)
//   binding 0 : array<BvhNode>     — the flat BVH tree
//   binding 1 : array<BvhTriangle> — world-space triangles in BVH leaf order
//   binding 2 : array<Material>    — PBR materials indexed by triangle
// Group 1 — per-frame data (updated every frame)
//   binding 0 : Camera uniform     — camera position + pre-scaled basis
//   binding 1 : FrameUniforms      — output width, height, sample index
//   binding 2 : array<u32>         — output RGBA8 pixels, row-major

// ============================================================================
// Structs — byte layouts MUST match the corresponding Rust #[repr(C)] types.
// ============================================================================

// Matches accel::bvh::BvhNode (32 bytes).
// vec3<f32> alignment = 16 in WGSL; the explicit padding u32s after each
// vec3 happen to land on the same offsets as the Rust repr(C) layout.
//   offset  0 : aabb_min        (12 bytes)
//   offset 12 : right_or_first  ( 4 bytes)  ← sits in the vec3's 4-byte tail gap
//   offset 16 : aabb_max        (12 bytes)
//   offset 28 : count           ( 4 bytes)
struct BvhNode {
    aabb_min:       vec3<f32>,
    right_or_first: u32,
    aabb_max:       vec3<f32>,
    count:          u32,
}

// Matches accel::bvh::BvhTriangle (64 bytes).
// Three back-to-back [f32; 3] arrays are packed at 12-byte stride in Rust.
// WGSL vec3<f32> has 16-byte alignment, so consecutive vec3s would drift from
// the Rust offsets. We flatten to individual f32 scalars to match exactly.
//   offsets  0..35  : v0x/y/z, v1x/y/z, v2x/y/z (nine f32s, 4 bytes each)
//   offsets 36..51  : i0, i1, i2, material_index (four u32s)
//   offsets 52..63  : _p0, _p1, _p2              (padding)
struct BvhTriangle {
    v0x: f32, v0y: f32, v0z: f32,
    v1x: f32, v1y: f32, v1z: f32,
    v2x: f32, v2y: f32, v2z: f32,
    i0: u32, i1: u32, i2: u32,
    material_index: u32,
    _p0: u32, _p1: u32, _p2: u32,
}

// Matches scene::material::Material (64 bytes).
// albedo (vec4, 16 bytes) lands cleanly. For emissive (3 floats after albedo)
// we use individual scalars so that metallic lands at offset 28, matching the
// Rust layout (a vec3 in WGSL would push metallic to offset 32).
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
    _p0: u32, _p1: u32, _p2: u32,
}

// Matches camera::CameraUniform (64 bytes).
// Each vec3 is followed by an explicit f32 pad to fill the 16-byte slot,
// giving the same offsets as the Rust struct.
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
    sample_index: u32,  // reserved for accumulation (Phase 4)
    _pad:         u32,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read>       bvh_nodes: array<BvhNode>;
@group(0) @binding(1) var<storage, read>       bvh_tris:  array<BvhTriangle>;
@group(0) @binding(2) var<storage, read>       materials: array<Material>;

@group(1) @binding(0) var<uniform>             camera:    Camera;
@group(1) @binding(1) var<uniform>             frame:     FrameUniforms;
@group(1) @binding(2) var<storage, read_write> output:    array<u32>;

// ============================================================================
// Constants
// ============================================================================

const T_MIN: f32 = 0.0001;  // minimum hit distance; avoids self-intersection
const T_MAX: f32 = 1.0e30;  // effective far plane

// A fixed directional light pointing roughly up-right-forward.
// normalize(vec3(1, 2, 1)) precomputed.
const LIGHT_DIR: vec3<f32> = vec3<f32>(0.4082, 0.8165, 0.4082);

const SKY_HORIZON: vec3<f32> = vec3<f32>(1.00, 1.00, 1.00);  // white horizon
const SKY_ZENITH:  vec3<f32> = vec3<f32>(0.49, 0.70, 1.00);  // sky blue

// ============================================================================
// Ray–AABB intersection (slab method)
// ============================================================================

// Returns true if the ray (orig, inv_dir) intersects the AABB within
// (T_MIN, t_max). inv_dir = 1.0 / dir — precomputed by the caller so this
// function does only multiplies, no divides.
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

// Returns vec3(t, u, v) on a hit, or vec3(-1, 0, 0) on a miss.
// Back-face culling: triangles whose front face points away from the ray are
// skipped (det < epsilon). t must be in (T_MIN, t_max) to count.
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
    if det < 1e-8 { return vec3(-1.0, 0.0, 0.0); }

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

// The result of a BVH traversal: the closest triangle hit along the ray.
// hit == 1 means a hit was found; hit == 0 means the ray escaped the scene.
// (bool is avoided here because WGSL bool members in structs can surprise
// some WebGPU implementations; u32 is unambiguous.)
struct HitRecord {
    t:       f32,
    tri_idx: u32,
    u:       f32,
    v:       f32,
    hit:     u32,
}

// Iterative depth-first BVH traversal with an explicit stack. Returns the
// closest intersection along (orig, dir, inv_dir), or hit=0 if none.
//
// The stack avoids GPU recursion (not allowed in WGSL). 64 entries is enough
// for any reasonably balanced tree built over real-world scene geometry.
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
            // Leaf: test every triangle in the leaf.
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
            // Internal: push right child first so left is processed first
            // (left child is always at idx + 1 in pre-order layout).
            stack[sp] = node.right_or_first;
            sp += 1;
            stack[sp] = idx + 1u;
            sp += 1;
        }
    }

    return result;
}

// ============================================================================
// Shading helpers
// ============================================================================

// Sky radiance for rays that miss all geometry: a gradient from a white
// horizon to a blue zenith, as in "Ray Tracing in One Weekend".
fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let t = 0.5 * (dir.y + 1.0);
    return mix(SKY_HORIZON, SKY_ZENITH, t);
}

// Pack a linear-light RGB colour into a RGBA8 u32 (R in low bits, A=255).
// Applies a gamma-2.0 approximation (sqrt) before quantising to 8 bits.
fn pack_rgba8(linear: vec3<f32>) -> u32 {
    let enc = sqrt(clamp(linear, vec3(0.0), vec3(1.0)));
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

    // Pixel centre in [0, 1]. v = 0 is the top row (matches Camera::ray).
    let u = (f32(gid.x) + 0.5) / f32(w);
    let v = (f32(gid.y) + 0.5) / f32(h);

    // Generate primary ray, mirroring the Camera::ray() formula on the CPU.
    let orig    = camera.position;
    let dir_raw = camera.fwd
                + (2.0 * u - 1.0) * camera.rgt_scaled
                + (1.0 - 2.0 * v) * camera.up_scaled;
    let dir     = normalize(dir_raw);
    let inv_dir = vec3(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);

    let hit = traverse_bvh(orig, dir, inv_dir);

    var color: vec3<f32>;
    if hit.hit == 0u {
        color = sky_color(dir);
    } else {
        let tri = bvh_tris[hit.tri_idx];
        let mat = materials[tri.material_index];

        // Flat face normal from vertex positions. Phase 4 will interpolate
        // smooth normals using the stored vertex indices (i0/i1/i2).
        let v0     = vec3(tri.v0x, tri.v0y, tri.v0z);
        let v1     = vec3(tri.v1x, tri.v1y, tri.v1z);
        let v2     = vec3(tri.v2x, tri.v2y, tri.v2z);
        let normal = normalize(cross(v1 - v0, v2 - v0));

        // Lambertian shading: ambient + diffuse from the fixed directional light.
        let n_dot_l  = max(dot(normal, LIGHT_DIR), 0.0);
        let albedo   = mat.albedo.rgb;
        let emissive = vec3(mat.emissive_r, mat.emissive_g, mat.emissive_b);

        color = albedo * (0.05 + 0.95 * n_dot_l) + emissive;
    }

    output[gid.y * w + gid.x] = pack_rgba8(color);
}
