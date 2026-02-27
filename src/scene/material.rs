/// Describes how a surface looks and interacts with light.
///
/// This follows the "Physically Based Rendering" (PBR) model, specifically
/// the metallic/roughness workflow used by GLTF. PBR materials are designed
/// to behave consistently under different lighting conditions and to match
/// the way real-world surfaces reflect light.
///
/// Every field has both a scalar value and an optional texture. If a texture
/// is present, it overrides the scalar for that property, allowing fine-grained
/// variation across the surface (e.g. a surface that is rough in some areas
/// and smooth in others). If no texture is present, the scalar applies uniformly.
pub struct Material {
    /// The base color of the surface, as an RGBA value. For non-metals this is
    /// essentially the "paint color". For metals it tints the reflections.
    /// Alpha (the fourth component) controls transparency.
    pub albedo: [f32; 4],

    /// How much light this surface emits on its own, independent of any external
    /// lighting. Non-zero values make the geometry act as a light source —
    /// this is how area lights are represented in path tracing. A value of
    /// [0, 0, 0] means the surface emits no light.
    pub emissive: [f32; 3],

    /// How "metal-like" the surface is, from 0.0 (fully dielectric, like plastic
    /// or wood) to 1.0 (fully metallic, like polished steel). Metals reflect
    /// light differently from non-metals: they absorb most colors and reflect
    /// tinted highlights.
    pub metallic: f32,

    /// How rough or smooth the surface is, from 0.0 (perfectly mirror-smooth)
    /// to 1.0 (completely matte/diffuse). Rough surfaces scatter reflected
    /// light in many directions; smooth surfaces produce sharp reflections.
    pub roughness: f32,

    // Texture indices into Scene::textures. A value of -1 means no texture is
    // present for that property — the corresponding scalar value is used instead.

    /// Texture that provides per-pixel albedo color.
    pub albedo_texture: i32,

    /// Normal map texture. Instead of storing geometry detail in extra triangles,
    /// a normal map encodes surface bumpiness as colored pixels, each pixel
    /// representing a small offset to the surface normal. This gives the
    /// appearance of fine detail (scratches, bumps, fabric weave) at almost
    /// no extra geometry cost.
    pub normal_texture: i32,

    /// Combined metallic (blue channel) and roughness (green channel) texture.
    /// GLTF packs both into one image to save memory.
    pub metallic_roughness_texture: i32,

    /// Texture that provides per-pixel emissive color, allowing parts of a
    /// surface to glow while others remain dark.
    pub emissive_texture: i32,
}

impl Default for Material {
    /// The GLTF "default material": a plain white, non-metallic, mid-roughness
    /// surface with no textures and no emission. Used for any primitive in a
    /// GLTF file that doesn't explicitly assign a material.
    fn default() -> Self {
        Self {
            albedo:                     [1.0, 1.0, 1.0, 1.0],
            emissive:                   [0.0, 0.0, 0.0],
            metallic:                   0.0,
            roughness:                  0.5,
            albedo_texture:             -1,
            normal_texture:             -1,
            metallic_roughness_texture: -1,
            emissive_texture:           -1,
        }
    }
}
