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
//   – All four texture maps applied: albedo, normal, metallic/roughness, emissive.
//   – TBN normal mapping via stored vertex tangents (GLTF convention: tangent.w
//     gives handedness; bitangent = cross(N, T) * w).
//   – One directional "sun" light + sky-gradient ambient for convincing shading.
//   – GGX NDF specular + Schlick Fresnel (no G2/V term — fast and close enough).
//   – No shadows, no reflections, no refraction — those are the path tracer's job.
//   – Reinhard tonemap + gamma so colours look reasonable.
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

/// Per-instance data: model transform + material index.
/// Must match GpuInstance in raster.rs (80 bytes, multiple of 16).
struct InstanceData {
    transform: mat4x4<f32>,  // local → world space (64 B)
    mat_index: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
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

@group(0) @binding(0) var<uniform>       camera:    RasterCamera;
@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;
@group(0) @binding(2) var<storage, read> materials: array<Material>;
/// Flat RGBA8 pixel data for all textures. One u32 per pixel: R in low byte,
/// A in high byte. Same packing as the path tracer's tex_data buffer.
@group(0) @binding(3) var<storage, read> tex_data:  array<u32>;
/// Per-texture metadata (pixel offset + dimensions). Indexed by material texture index.
@group(0) @binding(4) var<storage, read> tex_info:  array<TexInfo>;


// =============================================================================
// Texture sampling
// =============================================================================

/// Samples a packed RGBA8 texture from the flat storage buffer.
///
/// Matches `sample_texture()` in trace.wgsl. Returns (1,1,1,1) for invalid or
/// absent textures (idx < 0), so multiplying by the result is always safe.
///
/// UV coordinates are wrapped with fract() so values outside [0,1] tile
/// rather than clamp — standard GLTF REPEAT wrap mode.
fn sample_tex(idx: i32, uv: vec2<f32>) -> vec4<f32> {
    if (idx < 0) { return vec4<f32>(1.0); }
    let info = tex_info[u32(idx)];
    if (info.width == 0u || info.height == 0u) { return vec4<f32>(1.0); }

    let uf  = fract(uv.x);
    let vf  = fract(uv.y);
    let px  = min(u32(uf * f32(info.width)),  info.width  - 1u);
    let py  = min(u32(vf * f32(info.height)), info.height - 1u);
    let raw = tex_data[info.offset + py * info.width + px];

    return vec4<f32>(
        f32( raw        & 0xFFu) / 255.0,
        f32((raw >>  8u) & 0xFFu) / 255.0,
        f32((raw >> 16u) & 0xFFu) / 255.0,
        f32((raw >> 24u) & 0xFFu) / 255.0,
    );
}


// =============================================================================
// Vertex shader
// =============================================================================

struct VertexInput {
    @location(0) position: vec3<f32>,  // offset 0  (from Vertex struct)
    @location(1) normal:   vec3<f32>,  // offset 12
    @location(2) uv:       vec2<f32>,  // offset 24
    @location(3) tangent:  vec4<f32>,  // offset 32 — xyz tangent dir, w handedness

    @builtin(instance_index) instance_id: u32,
}

struct VertexOutput {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0) world_pos:          vec3<f32>,
    @location(1) world_normal:       vec3<f32>,
    @location(2) uv:                 vec2<f32>,
    @location(3) mat_index:          u32,
    @location(4) world_tangent:      vec4<f32>,  // xyz = world-space tangent, w = handedness
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let inst = instances[in.instance_id];

    // Transform position: local → world → camera → clip.
    let world_pos4 = inst.transform * vec4<f32>(in.position, 1.0);
    let clip_pos   = camera.proj * camera.view * world_pos4;

    // Upper-left 3×3 of the model matrix, used to transform direction vectors.
    // For non-uniform scaling, the correct normal transform is the transpose of
    // the inverse, but for the preview pass this approximation looks fine.
    let m3 = mat3x3<f32>(
        inst.transform[0].xyz,
        inst.transform[1].xyz,
        inst.transform[2].xyz,
    );
    let world_normal  = normalize(m3 * in.normal);
    // Tangent transforms the same way as normals for direction vectors.
    let world_tangent = vec4<f32>(normalize(m3 * in.tangent.xyz), in.tangent.w);

    var out: VertexOutput;
    out.clip_pos    = clip_pos;
    out.world_pos   = world_pos4.xyz;
    out.world_normal = world_normal;
    out.uv          = in.uv;
    out.mat_index   = inst.mat_index;
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

// Sun direction — kept consistent with trace.wgsl's SUN_DIR.
const SUN_DIR:   vec3<f32> = vec3<f32>(0.4,  0.9, 0.3);
const SUN_COLOR: vec3<f32> = vec3<f32>(4.0,  3.8, 3.0);  // warm white

// Sky gradient colours for hemispherical ambient.
// Blend from warm ground to cool blue zenith based on the surface normal's Y.
const SKY_ZENITH:  vec3<f32> = vec3<f32>(0.25, 0.45, 0.75);  // blue sky
const SKY_HORIZON: vec3<f32> = vec3<f32>(0.55, 0.65, 0.75);  // pale horizon
const GROUND_COL:  vec3<f32> = vec3<f32>(0.15, 0.13, 0.10);  // warm dark ground

const PI: f32 = 3.14159265358979;

/// Hemispherical sky ambient — varies the ambient colour based on the surface
/// normal direction. Surfaces facing up get blue sky; facing down get warm
/// ground; horizontal surfaces get the horizon colour.
///
/// This is a much better approximation of environment lighting than a flat
/// ambient constant, and costs one mix() call.
fn sky_ambient(N: vec3<f32>) -> vec3<f32> {
    let t = N.y * 0.5 + 0.5;  // maps [-1, 1] → [0, 1]: down=0, up=1
    let sky = mix(SKY_HORIZON, SKY_ZENITH, max(N.y, 0.0));
    return mix(GROUND_COL, sky, t);
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
    let L = normalize(SUN_DIR);
    let V = normalize(camera.cam_pos - in.world_pos);
    let H = normalize(L + V);

    let n_dot_l = max(dot(N, L), 0.0);
    let n_dot_h = max(dot(N, H), 0.0);
    let h_dot_v = max(dot(H, V), 0.0);

    // ---- Diffuse (Lambertian) ----
    // Metals have no diffuse response; their albedo colour goes into F0 instead.
    let diffuse = albedo * (1.0 - metallic) * n_dot_l * SUN_COLOR;

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
    let specular = (D * F * n_dot_l * SUN_COLOR) * 0.25;

    // ---- Hemispherical ambient ----
    // Use sky colour on the shading normal so surfaces facing upward get blue
    // sky and surfaces facing down get warm ground — much more convincing than
    // a flat constant, at no extra texture lookups.
    //
    // Occlusion texture (R channel): 0 = fully occluded, 1 = fully exposed.
    // Multiplying ambient by this value darkens crevices and cavities where
    // indirect light cannot easily reach. Only ambient is affected — direct
    // sun light is not occluded by this baked term.
    let occlusion = sample_tex(mat.occlusion_texture, uv).r;
    let ambient = albedo * sky_ambient(N) * occlusion;

    // ---- Emissive ----
    let emissive = mat.emissive * emit_samp;

    // ---- Glass hint ----
    // Real refraction needs the path tracer. In preview, dim transmissive
    // surfaces so you can at least tell they're glassy.
    let opacity = 1.0 - mat.transmission * 0.7;

    var color = (diffuse + specular + ambient + emissive) * opacity;

    // Reinhard tonemap + sRGB gamma encode. The output texture is Rgba8Unorm
    // (linear storage), so we manually apply gamma here.
    color = color / (color + vec3<f32>(1.0));
    color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
