use super::texture::Texture;

pub struct Environment {
    pub texture: Option<Texture>, // HDRI environment map
    pub tint:    [f32; 3],
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            texture: None,
            tint:    [1.0, 1.0, 1.0],
        }
    }
}
