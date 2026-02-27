pub struct Material {
    pub albedo:    [f32; 4], // base color + alpha
    pub emissive:  [f32; 3],
    pub metallic:  f32,
    pub roughness: f32,

    // Texture indices into Scene::textures; -1 means absent, use scalar fallback.
    pub albedo_texture:             i32,
    pub normal_texture:             i32,
    pub metallic_roughness_texture: i32,
    pub emissive_texture:           i32,
}

impl Default for Material {
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
