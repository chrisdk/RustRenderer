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
///
/// `#[repr(C)]` and `bytemuck::Pod` allow zero-copy GPU upload. The struct is
/// padded to 64 bytes because WGSL requires all array-element structs in
/// storage buffers to be a multiple of 16 bytes in size.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
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

    /// Index of refraction for transmissive (glass-like) materials.
    /// From the GLTF `KHR_materials_ior` extension; defaults to 1.5
    /// (typical crown glass) when the extension is absent.
    ///
    /// Only meaningful when `transmission > 0`. Determines how much the
    /// light ray bends as it crosses the surface (Snell's law) and how
    /// much is reflected vs refracted (Fresnel equations).
    pub ior: f32,

    /// Fraction of light that passes through the surface rather than being
    /// absorbed or scattered. From the GLTF `KHR_materials_transmission`
    /// extension; 0.0 for opaque surfaces, 1.0 for fully transmissive glass.
    ///
    /// When > 0 the shader fires a refracted ray through the surface rather
    /// than scattering diffusely, enabling caustics and colour-accurate glass.
    pub transmission: f32,

    /// Padding to reach 64 bytes (required multiple of 16 for WGSL storage
    /// buffer array elements). Always zero.
    pub _pad: u32,
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
            ior:                        1.5,
            transmission:               0.0,
            _pad:                       0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    use std::mem::{offset_of, size_of};

    /// Material is uploaded directly to a GPU storage buffer. If the layout
    /// drifts from what the WGSL struct expects, every surface in the scene
    /// will be shaded with garbage values — and it'll be very confusing.
    #[test]
    fn test_material_gpu_layout() {
        assert_eq!(size_of::<Material>(),                             64);
        assert_eq!(offset_of!(Material, albedo),                      0);
        assert_eq!(offset_of!(Material, emissive),                   16);
        assert_eq!(offset_of!(Material, metallic),                   28);
        assert_eq!(offset_of!(Material, roughness),                  32);
        assert_eq!(offset_of!(Material, albedo_texture),             36);
        assert_eq!(offset_of!(Material, normal_texture),             40);
        assert_eq!(offset_of!(Material, metallic_roughness_texture), 44);
        assert_eq!(offset_of!(Material, emissive_texture),           48);
        assert_eq!(offset_of!(Material, ior),                        52);
        assert_eq!(offset_of!(Material, transmission),               56);
        assert_eq!(offset_of!(Material, _pad),                       60);
        let _: &[u8] = bytemuck::bytes_of(&Material::zeroed());
    }
}
