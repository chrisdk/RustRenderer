use super::texture::Texture;

/// The background and ambient lighting of the scene, provided by an HDRI image.
///
/// An HDRI (High Dynamic Range Image) environment map is a 360° panoramic
/// photograph of a real or synthetic environment — a sky, a studio, an outdoor
/// scene — stored with enough brightness range to capture both dim shadows and
/// intense highlights. In path tracing, rays that escape the scene geometry
/// without hitting anything are aimed at this panorama to sample the incoming
/// light from that direction.
///
/// This gives scenes realistic ambient lighting and reflections "for free":
/// a shiny metal ball will reflect the sky, indoor objects will be softly lit
/// by a distant window, and so on, all driven by the single HDRI image.
pub struct Environment {
    /// The HDRI panorama. If None, the renderer falls back to a solid background
    /// color derived from the tint.
    pub texture: Option<Texture>,

    /// A color multiplier applied to all environment light. Defaults to white
    /// [1, 1, 1] (no change). Can be used to dim the environment or shift its
    /// color temperature without replacing the texture.
    pub tint: [f32; 3],
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            texture: None,
            tint:    [1.0, 1.0, 1.0],
        }
    }
}
