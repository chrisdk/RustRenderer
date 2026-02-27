pub mod environment;
pub mod geometry;
pub mod material;
pub mod texture;

use geometry::{Mesh, Vertex};
use material::Material;
use texture::Texture;

pub use environment::Environment;

pub struct Scene {
    pub vertices:    Vec<Vertex>,
    pub indices:     Vec<u32>,
    pub meshes:      Vec<Mesh>,
    pub materials:   Vec<Material>,
    pub textures:    Vec<Texture>,
    pub environment: Option<Environment>,
}

#[derive(Debug)]
pub enum SceneError {
    Gltf(gltf::Error),
    MissingAttribute(&'static str),
}

impl From<gltf::Error> for SceneError {
    fn from(e: gltf::Error) -> Self {
        SceneError::Gltf(e)
    }
}

impl Scene {
    pub fn from_gltf(data: &[u8]) -> Result<Self, SceneError> {
        let (document, buffers, images) = gltf::import_slice(data)?;

        // Textures are indexed by image index, so build this first.
        let textures: Vec<Texture> = images.iter()
            .map(Texture::from_gltf_image)
            .collect();

        // Collect explicit materials. The GLTF default material (index = None)
        // is appended lazily below if any primitive references it.
        let mut materials: Vec<Material> = document.materials()
            .map(|mat| {
                let pbr = mat.pbr_metallic_roughness();
                Material {
                    albedo:    pbr.base_color_factor(),
                    emissive:  mat.emissive_factor(),
                    metallic:  pbr.metallic_factor(),
                    roughness: pbr.roughness_factor(),
                    albedo_texture: pbr.base_color_texture()
                        .map(|i| i.texture().source().index() as i32)
                        .unwrap_or(-1),
                    normal_texture: mat.normal_texture()
                        .map(|i| i.texture().source().index() as i32)
                        .unwrap_or(-1),
                    metallic_roughness_texture: pbr.metallic_roughness_texture()
                        .map(|i| i.texture().source().index() as i32)
                        .unwrap_or(-1),
                    emissive_texture: mat.emissive_texture()
                        .map(|i| i.texture().source().index() as i32)
                        .unwrap_or(-1),
                }
            })
            .collect();

        let default_material_index = materials.len() as u32;
        let mut used_default_material = false;

        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices:  Vec<u32>   = Vec::new();
        let mut meshes:   Vec<Mesh>  = Vec::new();

        for mesh in document.meshes() {
            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));

                let vertex_start = vertices.len() as u32;
                let first_index  = indices.len() as u32;

                let positions: Vec<[f32; 3]> = reader
                    .read_positions()
                    .ok_or(SceneError::MissingAttribute("POSITION"))?
                    .collect();

                let vertex_count = positions.len();

                let normals: Vec<[f32; 3]> = reader
                    .read_normals()
                    .map(|it| it.collect())
                    .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; vertex_count]);

                let uvs: Vec<[f32; 2]> = reader
                    .read_tex_coords(0)
                    .map(|it| it.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0]; vertex_count]);

                let tangents: Vec<[f32; 4]> = reader
                    .read_tangents()
                    .map(|it| it.collect())
                    .unwrap_or_else(|| vec![[1.0, 0.0, 0.0, 1.0]; vertex_count]);

                vertices.extend((0..vertex_count).map(|i| Vertex {
                    position: positions[i],
                    normal:   normals[i],
                    uv:       uvs[i],
                    tangent:  tangents[i],
                }));

                let index_count = match reader.read_indices() {
                    Some(iter) => {
                        let raw: Vec<u32> = iter.into_u32().collect();
                        let count = raw.len() as u32;
                        indices.extend(raw.iter().map(|&i| i + vertex_start));
                        count
                    }
                    None => {
                        // Non-indexed primitive: generate sequential indices.
                        let count = vertex_count as u32;
                        indices.extend((0..count).map(|i| i + vertex_start));
                        count
                    }
                };

                let material_index = match primitive.material().index() {
                    Some(i) => i as u32,
                    None => {
                        used_default_material = true;
                        default_material_index
                    }
                };

                meshes.push(Mesh { first_index, index_count, material_index });
            }
        }

        if used_default_material {
            materials.push(Material::default());
        }

        Ok(Scene {
            vertices,
            indices,
            meshes,
            materials,
            textures,
            environment: None,
        })
    }
}
