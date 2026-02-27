// Path-tracing compute shader.
//
// One GPU thread per pixel. Shoots a jittered primary ray, finds the closest
// triangle via iterative BVH traversal, shades with a Lambertian BRDF and a
// fixed directional sky light, then accumulates the result into a float buffer.
//
// On each dispatch the sample_index field says which sample number this is.
// When sample_index == 0 the accumulator is reset; otherwise the new sample is
// added to the running sum and the average is written to the output buffer.
// Calling the shader N times with sample_index 0..N-1 gives an N-sample
// anti-aliased image.  The pixel jitter (a sub-pixel offset drawn from a
// PCG hash of the pixel coords + sample_index) ensures every sample differs,
// so the average converges to a smooth image rather than the same sharp-edged
// frame repeated N times.
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

// Matches accel::bvh::BvhTriangle (128 bytes).
//
// The Rust struct uses [f32; 3] for positions, normals, and [f32; 2] for UVs.
// WGSL vec3<f32> has 16-byte alignment, which would drift from the Rust offsets,
// so every field is spelled out as individual f32/u32 scalars instead.
//
//   offsets   0..35  : v0/v1/v2 positions (nine f32s)
//   offsets  36..51  : i0, i1, i2, material_index (four u32s)
//   offsets  52..87  : n0/n1/n2 normals (nine f32s)
//   offsets  88..111 : uv0/uv1/uv2 (six f32s)
//   offsets 112..127 : padding (four u32s)
struct BvhTriangle {
    // World-space vertex positions (for intersection)
    v0x: f32, v0y: f32, v0z: f32,
    v1x: f32, v1y: f32, v1z: f32,
    v2x: f32, v2y: f32, v2z: f32,
    // Vertex indices and material
    i0: u32, i1: u32, i2: u32,
    material_index: u32,
    // World-space vertex normals (for smooth shading)
    n0x: f32, n0y: f32, n0z: f32,
    n1x: f32, n1y: f32, n1z: f32,
    n2x: f32, n2y: f32, n2z: f32,
    // UV texture coordinates
    uv0u: f32, uv0v: f32,
    uv1u: f32, uv1v: f32,
    uv2u: f32, uv2v: f32,
    // Padding to 128 bytes
    _p0: u32, _p1: u32, _p2: u32, _p3: u32,
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

// Matches renderer::gpu::GpuTextureInfo (16 bytes).
struct TexInfo {
    /// Pixel index into tex_data where this texture's pixels start.
    offset: u32,
    width:  u32,
    height: u32,
    _pad:   u32,
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
    sample_index: u32,
    _pad:         u32,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read>       bvh_nodes: array<BvhNode>;
@group(0) @binding(1) var<storage, read>       bvh_tris:  array<BvhTriangle>;
@group(0) @binding(2) var<storage, read>       materials: array<Material>;
@group(0) @binding(3) var<storage, read>       tex_data:  array<u32>;   // RGBA8 packed pixels
@group(0) @binding(4) var<storage, read>       tex_info:  array<TexInfo>;

@group(1) @binding(0) var<uniform>             camera:  Camera;
@group(1) @binding(1) var<uniform>             frame:   FrameUniforms;
@group(1) @binding(2) var<storage, read_write> output:  array<u32>;
@group(1) @binding(3) var<storage, read_write> accum:   array<vec4<f32>>;

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
// PCG random number generator
// ============================================================================

// PCG-RXS-M-XS: a minimal but high-quality 32-bit hash. Much better than a
// plain LCG for use as a noise source — produces no visible patterns.
// Reference: O'Neill, "PCG: A Family of Simple Fast Space-Efficient
// Statistically Good Algorithms for Random Number Generation", 2014.
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Advance the seed in place and return a float in [0, 1).
fn rand_f32(seed: ptr<function, u32>) -> f32 {
    *seed = pcg(*seed);
    return f32(*seed) / 4294967296.0;
}

// ============================================================================
// Texture sampling
// ============================================================================

// Sample a texture by index using nearest-neighbour filtering with UV wrap.
//
// tex_idx < 0 means "no texture" — returns a white vec4 so the caller can
// multiply it against the material's scalar factor without changing it.
//
// UV coordinates are wrapped (modulo 1.0) so values outside [0, 1] repeat
// the texture rather than clamping to the edge, which is the GLTF default.
fn sample_texture(tex_idx: i32, uv: vec2<f32>) -> vec4<f32> {
    if tex_idx < 0 {
        return vec4(1.0, 1.0, 1.0, 1.0);
    }

    let info = tex_info[u32(tex_idx)];

    // fract(x) = x - floor(x), which wraps any value into [0, 1).
    let uf = fract(uv.x);
    let vf = fract(uv.y);

    // Convert to integer pixel coordinates, clamped so we never read past the
    // last row/column of a texture when uf/vf rounds up to exactly 1.0.
    let px = min(u32(uf * f32(info.width)),  info.width  - 1u);
    let py = min(u32(vf * f32(info.height)), info.height - 1u);

    // RGBA8 pixels are packed as one u32: R in byte 0, A in byte 3.
    let packed = tex_data[info.offset + py * info.width + px];

    return vec4(
        f32( packed        & 0xFFu) / 255.0,  // R
        f32((packed >>  8u) & 0xFFu) / 255.0,  // G
        f32((packed >> 16u) & 0xFFu) / 255.0,  // B
        f32((packed >> 24u) & 0xFFu) / 255.0,  // A
    );
}

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

    let px  = gid.x;
    let py  = gid.y;
    let idx = py * w + px;

    // ── Per-sample RNG seed ───────────────────────────────────────────────────
    //
    // XOR three independent terms so neighbouring pixels and successive samples
    // all start from unrelated seeds. The PCG step that follows scrambles even
    // poor seeds into high-quality noise.
    var seed = (px * 1973u) ^ (py * 9277u) ^ (frame.sample_index * 26699u + 1u);

    // ── Jittered primary ray ──────────────────────────────────────────────────
    //
    // A uniform random offset within the pixel footprint turns the single-sample
    // aliased result into a stochastic estimate of the true box-filter integral.
    // Accumulate enough samples and the average converges to a sharp, clean image.
    let jx = rand_f32(&seed) - 0.5;  // jitter in [-0.5, 0.5] pixels
    let jy = rand_f32(&seed) - 0.5;

    let u = (f32(px) + 0.5 + jx) / f32(w);
    let v = (f32(py) + 0.5 + jy) / f32(h);

    // Generate the primary ray, mirroring Camera::ray() on the CPU.
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

        // ── Smooth normal ─────────────────────────────────────────────────────
        //
        // Interpolate the three pre-baked world-space vertex normals using the
        // hit's barycentric coordinates (u, v from Möller–Trumbore).
        //   w = weight for vertex 0 (= 1 − u − v)
        //   u = weight for vertex 1
        //   v = weight for vertex 2
        // The result is a per-fragment normal that smoothly follows the surface
        // curvature, removing the faceted look you'd get from a flat face normal.
        let bary_w = 1.0 - hit.u - hit.v;
        let n0 = vec3(tri.n0x, tri.n0y, tri.n0z);
        let n1 = vec3(tri.n1x, tri.n1y, tri.n1z);
        let n2 = vec3(tri.n2x, tri.n2y, tri.n2z);
        let normal = normalize(bary_w * n0 + hit.u * n1 + hit.v * n2);

        // ── UV interpolation ──────────────────────────────────────────────────
        let uv = bary_w * vec2(tri.uv0u, tri.uv0v)
               + hit.u  * vec2(tri.uv1u, tri.uv1v)
               + hit.v  * vec2(tri.uv2u, tri.uv2v);

        // ── Albedo ────────────────────────────────────────────────────────────
        //
        // The GLTF spec says the effective albedo is the base-color factor
        // multiplied component-wise by the base-color texture sample (if any).
        // Where no texture is present, sample_texture() returns white (1,1,1,1)
        // so the multiplication is a no-op.
        var albedo = mat.albedo.rgb * sample_texture(mat.albedo_tex, uv).rgb;

        // ── Emissive ──────────────────────────────────────────────────────────
        var emissive = vec3(mat.emissive_r, mat.emissive_g, mat.emissive_b)
                     * sample_texture(mat.emissive_tex, uv).rgb;

        // ── Lambertian shading ────────────────────────────────────────────────
        //
        // A single directional light gives a quick approximation of direct
        // illumination. Ambient term (0.05) prevents surfaces facing away from
        // the light from going completely black — a crude stand-in for indirect
        // illumination until multi-bounce path tracing is implemented.
        let n_dot_l = max(dot(normal, LIGHT_DIR), 0.0);
        color = albedo * (0.05 + 0.95 * n_dot_l) + emissive;
    }

    // ── Progressive accumulation ──────────────────────────────────────────────
    //
    // On sample 0 the accumulator is seeded with the first sample's value.
    // On subsequent samples the new value is added to the running sum, and the
    // output buffer gets the current average. This means the canvas updates after
    // every sample, giving a live "image forming" feel during a long render.
    let sample = vec4<f32>(color, 1.0);
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
