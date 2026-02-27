//! The scene representation: everything the renderer needs to know about
//! what is being rendered, independent of how it will be rendered.
//!
//! A scene is loaded from a GLTF file and then held in memory for the lifetime
//! of the rendering session. It is separated into two layers:
//!
//! - The **mesh library**: unique geometry (vertices and triangles) stored once,
//!   in local coordinate space.
//! - The **instance list**: a list of (mesh, world transform) pairs describing
//!   where each mesh actually appears in the scene. The same mesh can appear
//!   multiple times with different transforms (instancing).
//!
//! This separation avoids duplicating geometry data for repeated objects and
//! keeps the scene representation faithful to the original GLTF structure.
//! When the acceleration structure (BVH) is built, it will read the instance
//! list, apply each transform, and work entirely in world space.

pub mod environment;
pub mod geometry;
pub mod material;
pub mod texture;

use geometry::{Mesh, MeshInstance, Vertex};
use material::Material;
use texture::Texture;

pub use environment::Environment;

/// Everything the renderer needs to know about the scene.
///
/// This struct owns all scene data: geometry, materials, textures, and the
/// list of instances that describes where objects are placed in the world.
/// It is populated once at load time from a GLTF file and then treated as
/// immutable for the duration of the rendering session.
pub struct Scene {
    /// All vertices from all meshes, concatenated into one flat array.
    /// Individual meshes index into this array using their `first_index`
    /// and `index_count` fields.
    pub vertices: Vec<Vertex>,

    /// Triangle index data for all meshes, concatenated into one flat array.
    /// Every three consecutive entries form one triangle: they are indices
    /// into `vertices` identifying the triangle's three corners.
    pub indices: Vec<u32>,

    /// The mesh library: unique geometry definitions, stored in local space.
    /// A Mesh does not know where in the world it is; that is the job of
    /// MeshInstance.
    pub meshes: Vec<Mesh>,

    /// The instance list: every object that appears in the scene, as a
    /// (mesh index, world transform) pair. This is what determines where
    /// each piece of geometry is actually placed, rotated, and scaled.
    pub instances: Vec<MeshInstance>,

    /// All materials referenced by meshes in this scene.
    pub materials: Vec<Material>,

    /// All decoded textures referenced by materials in this scene.
    pub textures: Vec<Texture>,

    /// Optional environment map providing background lighting. If None,
    /// rays that escape the scene geometry hit a black background.
    pub environment: Option<Environment>,
}

/// Errors that can occur while loading a scene.
#[derive(Debug)]
pub enum SceneError {
    /// The GLTF file could not be parsed (malformed data, unsupported version, etc.).
    Gltf(gltf::Error),

    /// A mesh primitive is missing a required vertex attribute.
    MissingAttribute(&'static str),
}

impl From<gltf::Error> for SceneError {
    fn from(e: gltf::Error) -> Self {
        SceneError::Gltf(e)
    }
}

impl Scene {
    /// Loads a scene from raw GLTF or GLB bytes.
    ///
    /// GLTF (GL Transmission Format) is the standard interchange format for
    /// 3D scenes. It can come as a `.gltf` file (JSON + separate binary files)
    /// or as a self-contained `.glb` binary bundle. This function accepts either.
    ///
    /// Loading happens in two passes:
    ///
    /// **Pass 1** reads every mesh primitive in the GLTF mesh library and
    /// appends its vertices and indices to the scene's flat buffers. This
    /// populates `vertices`, `indices`, and `meshes`, and builds an internal
    /// mapping from GLTF mesh indices to our Mesh indices.
    ///
    /// **Pass 2** traverses the GLTF scene graph — a tree of nodes, each with
    /// a transform and an optional mesh reference — to produce the `instances`
    /// list. Each node that references a mesh becomes a MeshInstance carrying
    /// the node's accumulated world-space transform.
    pub fn from_gltf(data: &[u8]) -> Result<Self, SceneError> {
        let (document, buffers, images) = gltf::import_slice(data)?;

        let textures: Vec<Texture> = images.iter()
            .map(Texture::from_gltf_image)
            .collect();

        // Collect all materials defined in the GLTF file.
        // The GLTF "default material" (used by primitives that don't name one)
        // is not listed here — it gets appended lazily below if needed.
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

        // Reserve an index for the default material; only push it if used.
        let default_material_index = materials.len() as u32;
        let mut used_default_material = false;

        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices:  Vec<u32>   = Vec::new();
        let mut meshes:   Vec<Mesh>  = Vec::new();

        // ----------------------------------------------------------------
        // Pass 1: load geometry
        //
        // GLTF meshes are made of one or more "primitives" — each primitive
        // is an independent set of triangles with its own material. We map
        // each primitive to one of our Mesh structs.
        //
        // mesh_map[gltf_mesh_index] = list of our Mesh indices for that mesh's
        // primitives. Pass 2 uses this to look up which Mesh indices to
        // instantiate when it encounters a node that references a GLTF mesh.
        // ----------------------------------------------------------------
        let mut mesh_map: Vec<Vec<u32>> = Vec::new();

        for gltf_mesh in document.meshes() {
            let mut primitive_indices = Vec::new();

            for primitive in gltf_mesh.primitives() {
                // The reader provides typed iterators over each vertex attribute.
                let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));

                // Record where this primitive's data starts in the shared buffers.
                let vertex_start = vertices.len() as u32;
                let first_index  = indices.len() as u32;

                // POSITION is the only required attribute; everything else falls
                // back to a sensible default if absent.
                let positions: Vec<[f32; 3]> = reader
                    .read_positions()
                    .ok_or(SceneError::MissingAttribute("POSITION"))?
                    .collect();

                let vertex_count = positions.len();

                // Default normal: straight up. Better than crashing, though
                // geometry without normals will be lit incorrectly.
                let normals: Vec<[f32; 3]> = reader
                    .read_normals()
                    .map(|it| it.collect())
                    .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; vertex_count]);

                // Default UV: top-left corner of the texture.
                let uvs: Vec<[f32; 2]> = reader
                    .read_tex_coords(0)
                    .map(|it| it.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0]; vertex_count]);

                // Default tangent: pointing along the X axis, positive handedness.
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
                        // Offset indices so they point into the shared vertex array
                        // rather than the local start of this primitive's vertices.
                        indices.extend(raw.iter().map(|&i| i + vertex_start));
                        count
                    }
                    None => {
                        // Non-indexed primitive: vertices are used in order,
                        // so we synthesize sequential indices.
                        let count = vertex_count as u32;
                        indices.extend((0..count).map(|i| i + vertex_start));
                        count
                    }
                };

                let material_index = match primitive.material().index() {
                    Some(i) => i as u32,
                    None    => {
                        // This primitive uses the GLTF default material.
                        used_default_material = true;
                        default_material_index
                    }
                };

                let mesh_index = meshes.len() as u32;
                meshes.push(Mesh { first_index, index_count, material_index });
                primitive_indices.push(mesh_index);
            }

            mesh_map.push(primitive_indices);
        }

        if used_default_material {
            materials.push(Material::default());
        }

        // ----------------------------------------------------------------
        // Pass 2: traverse the scene graph
        //
        // A GLTF file contains a library of meshes (loaded above) and a scene
        // graph: a tree of nodes. Each node has a local transform and may
        // reference a mesh. Nodes can be nested — a chair mesh might be a child
        // of a room node, inheriting the room's position.
        //
        // We walk the tree recursively, accumulating the product of all parent
        // transforms on the way down. When we reach a node that has a mesh, we
        // emit a MeshInstance for each of its primitives carrying the full
        // world-space transform.
        // ----------------------------------------------------------------
        let mut instances: Vec<MeshInstance> = Vec::new();

        let gltf_scene = document.default_scene()
            .or_else(|| document.scenes().next());

        match gltf_scene {
            Some(s) => {
                for node in s.nodes() {
                    traverse_node(&node, glam::Mat4::IDENTITY, &mesh_map, &mut instances);
                }
            }
            None => {
                // Some GLTF files contain only a mesh library with no scene.
                // Fall back to placing every mesh at the world origin.
                for primitive_indices in &mesh_map {
                    for &mesh_index in primitive_indices {
                        instances.push(MeshInstance {
                            mesh_index,
                            transform: glam::Mat4::IDENTITY.to_cols_array_2d(),
                        });
                    }
                }
            }
        }

        Ok(Scene {
            vertices,
            indices,
            meshes,
            instances,
            materials,
            textures,
            environment: None,
        })
    }
}

/// Recursively walks a GLTF node and its children, emitting a MeshInstance
/// for every node that references a mesh.
///
/// `parent_transform` is the accumulated world-space transform of all ancestor
/// nodes. At each step we multiply the node's own local transform onto it,
/// so by the time we reach a leaf node the product is the full object-to-world
/// transform for anything attached to that node.
fn traverse_node(
    node: &gltf::Node,
    parent_transform: glam::Mat4,
    mesh_map: &[Vec<u32>],
    instances: &mut Vec<MeshInstance>,
) {
    // Combine this node's local transform with the inherited parent transform.
    // GLTF stores the local transform as a column-major 4×4 matrix.
    let local = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent_transform * local;

    if let Some(mesh) = node.mesh() {
        // A GLTF mesh can have multiple primitives (each with its own material).
        // Each primitive maps to one of our Mesh entries via mesh_map.
        for &mesh_index in &mesh_map[mesh.index()] {
            instances.push(MeshInstance {
                mesh_index,
                transform: world.to_cols_array_2d(),
            });
        }
    }

    // Recurse into child nodes, passing along the accumulated world transform.
    for child in node.children() {
        traverse_node(&child, world, mesh_map, instances);
    }
}
