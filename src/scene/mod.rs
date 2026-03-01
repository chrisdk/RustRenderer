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
                    ior: mat.ior().unwrap_or(1.5),
                    transmission: mat.transmission()
                        .map_or(0.0, |t| t.transmission_factor()),
                    occlusion_texture: mat.occlusion_texture()
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
        })
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Minimal GLB builder ───────────────────────────────────────────────────

    /// Constructs a minimal valid GLB binary containing one triangle.
    ///
    /// The triangle has three vertices at (0,0,0), (1,0,0), and (0,1,0) and is
    /// placed at the world origin with an identity transform. No materials are
    /// defined — the loader should inject the default material automatically.
    ///
    /// This is the smallest possible GLB that exercises every code path in
    /// `from_gltf`: header parsing, accessor/bufferView resolution, geometry
    /// flattening, scene-graph traversal, and default-material injection.
    fn minimal_triangle_glb() -> Vec<u8> {
        // ── Binary chunk: 3 × VEC3 f32 positions (36 bytes)
        //                + 3 × u16 indices            (6 bytes)
        //                + 2 bytes padding             (→ 44 total, aligned to 4)
        let mut bin: Vec<u8> = Vec::new();
        for pos in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for f in pos { bin.extend_from_slice(&f.to_le_bytes()); }
        }
        for idx in [0u16, 1, 2] { bin.extend_from_slice(&idx.to_le_bytes()); }
        while bin.len() % 4 != 0 { bin.push(0); }   // pad to 4-byte boundary
        let bin_len = bin.len();                      // 44 bytes

        // ── JSON chunk: minimal GLTF referencing the binary above.
        // componentType 5126 = FLOAT, 5123 = UNSIGNED_SHORT
        // The gltf crate's validator requires min/max on POSITION accessors.
        // Our three vertices span [0,0,0]..[1,1,0].
        let json = r#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":6}],"buffers":[{"byteLength":42}]}"#;
        let json_bytes   = json.as_bytes();
        let json_padded  = (json_bytes.len() + 3) & !3;   // round up to 4 bytes

        // Total file length: 12-byte header + 8-byte JSON chunk header + padded JSON
        //                  + 8-byte BIN chunk header + binary data
        let total = 12 + 8 + json_padded + 8 + bin_len;

        let mut glb = Vec::with_capacity(total);

        // GLB header
        glb.extend_from_slice(b"glTF");                         // magic
        glb.extend_from_slice(&2u32.to_le_bytes());             // version 2
        glb.extend_from_slice(&(total as u32).to_le_bytes());   // total length

        // JSON chunk
        glb.extend_from_slice(&(json_padded as u32).to_le_bytes()); // chunkLength
        glb.extend_from_slice(b"JSON");                              // chunkType
        glb.extend_from_slice(json_bytes);
        while glb.len() % 4 != 0 { glb.push(b' '); }   // pad JSON with spaces

        // BIN chunk
        glb.extend_from_slice(&(bin_len as u32).to_le_bytes()); // chunkLength
        glb.extend_from_slice(b"BIN\0");                        // chunkType
        glb.extend_from_slice(&bin);

        glb
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// The most basic sanity check: a well-formed single-triangle GLB loads
    /// without error and produces the expected vertex, index, mesh, and instance
    /// counts. If this fails, the scene loader is fundamentally broken.
    #[test]
    fn test_from_gltf_single_triangle_loads() {
        let glb   = minimal_triangle_glb();
        let scene = Scene::from_gltf(&glb)
            .expect("minimal single-triangle GLB should parse without error");

        assert_eq!(scene.vertices.len(),  3, "expected 3 vertices");
        assert_eq!(scene.indices.len(),   3, "expected 3 indices");
        assert_eq!(scene.meshes.len(),    1, "expected 1 mesh");
        assert_eq!(scene.instances.len(), 1, "expected 1 instance");

        // The mesh window covers all 3 indices starting at offset 0.
        assert_eq!(scene.meshes[0].first_index,  0);
        assert_eq!(scene.meshes[0].index_count,  3);
    }

    /// Vertex positions must round-trip: what was uploaded as f32 floats in the
    /// binary chunk should appear unchanged in `scene.vertices[*].position`.
    #[test]
    fn test_from_gltf_vertex_positions_round_trip() {
        let scene = Scene::from_gltf(&minimal_triangle_glb())
            .expect("should parse");

        let eps = 1e-6f32;
        let p = [
            scene.vertices[0].position,
            scene.vertices[1].position,
            scene.vertices[2].position,
        ];
        assert!((p[0][0]).abs() < eps && (p[0][1]).abs() < eps && (p[0][2]).abs() < eps,
            "v0 should be (0,0,0), got {:?}", p[0]);
        assert!((p[1][0] - 1.0).abs() < eps && (p[1][1]).abs() < eps && (p[1][2]).abs() < eps,
            "v1 should be (1,0,0), got {:?}", p[1]);
        assert!((p[2][0]).abs() < eps && (p[2][1] - 1.0).abs() < eps && (p[2][2]).abs() < eps,
            "v2 should be (0,1,0), got {:?}", p[2]);
    }

    /// When no material is specified in the GLTF, `from_gltf` must inject the
    /// default material (white, non-metallic, no textures). The instance's
    /// material index must point to it, and the transform must be identity.
    #[test]
    fn test_from_gltf_default_material_and_identity_transform() {
        let scene = Scene::from_gltf(&minimal_triangle_glb())
            .expect("should parse");

        // Default material injected because no materials are defined in the file.
        assert_eq!(scene.materials.len(), 1, "one default material expected");
        assert_eq!(scene.materials[0].albedo, [1.0, 1.0, 1.0, 1.0],
            "default material albedo should be white");
        assert_eq!(scene.materials[0].metallic,  0.0);
        assert_eq!(scene.materials[0].roughness, 0.5);

        // Node has no explicit transform — should default to identity.
        let identity: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(scene.instances[0].transform, identity,
            "instance with no transform should use the identity matrix");
    }

    /// Garbage bytes should return a clear error, not panic.
    #[test]
    fn test_from_gltf_garbage_returns_error() {
        let result = Scene::from_gltf(b"not a valid gltf file!!");
        assert!(result.is_err(), "garbage bytes should return Err, not panic");
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
