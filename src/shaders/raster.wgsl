// =============================================================================
// Rasterising preview shader — vertex + fragment
// =============================================================================
//
// This shader powers the *fast* interactive preview path. Path tracing is
// gorgeous but slow; rasterisation is approximate but runs at 60 fps. The
// goal here is "close enough to spin the model around and get your bearings"
// rather than "physically accurate". The path tracer handles that when the
// user clicks Render.
//
// Algorithm: single-pass PBR-ish approximation.
//   – One directional "sun" light at a fixed world-space direction.
//   – Cheap ambient term to fake indirect illumination (the "sky" colour).
//   – Lambertian diffuse + GGX specular, evaluated analytically (no Monte Carlo).
//   – No shadows, no reflections, no refraction — those belong to the path tracer.
//   – Transmissive surfaces (glass) rendered as partially transparent but dark
//     (just enough to distinguish them from opaque geometry in the preview).
//   – Reinhard tonemapping + gamma so colours look reasonable without HDR.
//
// The Material struct here must byte-for-byte match src/scene/material.rs.
// =============================================================================


// =============================================================================
// Bind group 0 — camera + per-instance data + material table
// =============================================================================

/// Camera matrices plus world-space position for specular highlights.
/// Rust struct: RasterCameraUniform (144 bytes).
struct RasterCamera {
    view:    mat4x4<f32>,  // world → camera space (offset 0, 64 B)
    proj:    mat4x4<f32>,  // camera space → clip space (offset 64, 64 B)
    cam_pos: vec3<f32>,    // world-space camera position (offset 128, 12 B)
    _pad:    f32,          // padding to 144 B (offset 140)
}

/// Per-instance data: model transform + material index.
/// One entry per MeshInstance, laid out in the same order as draw calls.
/// Must match GpuInstance in raster.rs (80 bytes, multiple of 16).
struct InstanceData {
    transform: mat4x4<f32>,  // local → world space (64 B)
    mat_index: u32,          // index into `materials` array (4 B)
    _p0: u32,                // padding
    _p1: u32,
    _p2: u32,
}

/// Material parameters, byte-for-byte identical to src/scene/material.rs.
/// 64 bytes, must remain in sync with the Rust struct.
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
    _pad:                       u32,         //                      (offset 60)
}

@group(0) @binding(0) var<uniform>       camera:    RasterCamera;
@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;
@group(0) @binding(2) var<storage, read> materials: array<Material>;


// =============================================================================
// Vertex shader
// =============================================================================

struct VertexInput {
    // Must match the wgpu::VertexBufferLayout in raster.rs, which is
    // derived from the Vertex struct (48 bytes: pos/normal/uv/tangent).
    @location(0) position: vec3<f32>,  // offset 0
    @location(1) normal:   vec3<f32>,  // offset 12
    @location(2) uv:       vec2<f32>,  // offset 24
    @location(3) tangent:  vec4<f32>,  // offset 32

    // The GPU fills this automatically with the zero-based draw instance index.
    @builtin(instance_index) instance_id: u32,
}

struct VertexOutput {
    @builtin(position) clip_pos:    vec4<f32>,
    @location(0)       world_pos:   vec3<f32>,
    @location(1)       world_normal: vec3<f32>,
    @location(2)       uv:          vec2<f32>,
    @location(3)       mat_index:   u32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let inst = instances[in.instance_id];

    // Transform position: local → world → camera → clip.
    let world_pos4 = inst.transform * vec4<f32>(in.position, 1.0);
    let clip_pos   = camera.proj * camera.view * world_pos4;

    // Transform normal to world space using the upper-left 3×3 of the model
    // matrix. For non-uniform scaling this isn't quite right (we should use
    // the transpose of the inverse), but for the preview it's close enough.
    // If the model has wildly non-uniform scale, normals will look a bit off —
    // the path tracer will fix it when you click Render.
    let m3 = mat3x3<f32>(
        inst.transform[0].xyz,
        inst.transform[1].xyz,
        inst.transform[2].xyz,
    );
    let world_normal = normalize(m3 * in.normal);

    var out: VertexOutput;
    out.clip_pos    = clip_pos;
    out.world_pos   = world_pos4.xyz;
    out.world_normal = world_normal;
    out.uv          = in.uv;
    out.mat_index   = inst.mat_index;
    return out;
}


// =============================================================================
// Fragment shader
// =============================================================================

// Sun direction — kept consistent with trace.wgsl's SUN_DIR so the rasteriser
// and path tracer roughly agree on where the light comes from.
const SUN_DIR:   vec3<f32> = vec3<f32>(0.4,  0.9, 0.3);   // not pre-normalised; we normalise below
const SUN_COLOR: vec3<f32> = vec3<f32>(4.0,  3.8, 3.0);   // white-ish, slightly warm
const AMBIENT:   vec3<f32> = vec3<f32>(0.12, 0.14, 0.18); // cool blue-grey sky ambient

const PI: f32 = 3.14159265358979;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let mat = materials[in.mat_index];

    let L = normalize(SUN_DIR);           // direction *toward* the light
    let N = normalize(in.world_normal);
    let V = normalize(camera.cam_pos - in.world_pos);  // direction *toward* the camera

    let albedo    = mat.albedo.rgb;
    let metallic  = mat.metallic;
    let roughness = max(mat.roughness, 0.04);  // avoid NaN in D_GGX when roughness = 0

    // ---- NdotL ----
    let n_dot_l = max(dot(N, L), 0.0);

    // ---- Diffuse (Lambertian) ----
    // Metals have no diffuse contribution; their albedo goes into the F0 term.
    let diffuse = albedo * (1.0 - metallic) * n_dot_l * SUN_COLOR;

    // ---- Specular (cheap GGX + Schlick) ----
    // We skip visibility (G2 Smith) for speed — the GGX normal distribution
    // function alone gives a good specular highlight shape.
    let H = normalize(L + V);
    let n_dot_h = max(dot(N, H), 0.0);
    let h_dot_v = max(dot(H, V), 0.0);

    let alpha  = roughness * roughness;
    let alpha2 = alpha * alpha;
    let d_denom = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    let D = alpha2 / (PI * d_denom * d_denom);  // GGX NDF

    // Schlick Fresnel. For dielectrics F0 ≈ 0.04; for metals F0 = albedo.
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let F  = f0 + (1.0 - f0) * pow(1.0 - h_dot_v, 5.0);

    // A crude 1/4 denominator approximation (skipping G2 and the π factor from D).
    let specular = (D * F * n_dot_l * SUN_COLOR) * 0.25;

    // ---- Ambient ----
    let ambient = albedo * AMBIENT;

    // ---- Emissive ----
    let emissive = mat.emissive;

    // ---- Glass hint ----
    // True refraction needs the path tracer. In preview, just dim transmissive
    // surfaces so you can tell they're glassy — better than rendering opaque plastic.
    let opacity = 1.0 - mat.transmission * 0.7;

    var color = (diffuse + specular + ambient + emissive) * opacity;

    // Reinhard tonemap + sRGB gamma. The output texture is Rgba8Unorm (linear),
    // so we bake gamma in the shader rather than relying on hardware sRGB.
    color = color / (color + vec3<f32>(1.0));
    color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
