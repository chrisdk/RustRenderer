/// A single point on the surface of a 3D object, bundled with all the data
/// the renderer needs at that point.
///
/// 3D meshes are made of triangles. Each triangle has three corners, and each
/// corner is a Vertex. Beyond just its position in space, a vertex carries
/// extra per-point data that the renderer uses to compute lighting and apply
/// textures correctly.
///
/// The `#[repr(C)]` attribute fixes the memory layout to match the order the
/// fields are declared, with no surprise padding inserted by the compiler.
/// Combined with `bytemuck::Pod`, this lets us cast a `&[Vertex]` directly
/// to `&[u8]` for zero-copy GPU buffer upload.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// The (x, y, z) position of this point in the mesh's local coordinate space.
    /// "Local space" means the coordinates are relative to the mesh's own origin,
    /// before any world transform (position, rotation, scale) has been applied.
    pub position: [f32; 3],

    /// A unit vector pointing perpendicularly away from the surface at this vertex.
    /// Normals tell the renderer which direction the surface is "facing", which
    /// determines how light bounces off it. Without correct normals a surface
    /// cannot be lit properly.
    pub normal: [f32; 3],

    /// 2D texture coordinates (also called "UV coordinates") for this vertex.
    /// UV values range from 0.0 to 1.0 and describe how a texture image is
    /// "wrapped" onto the surface — u goes left-to-right, v goes top-to-bottom.
    pub uv: [f32; 2],

    /// A vector that lies flat along the surface, perpendicular to the normal.
    /// Tangents are needed for normal mapping: a normal map encodes surface
    /// detail as vectors in "tangent space" (relative to the surface itself),
    /// and the tangent tells us how to orient that local space in the world.
    ///
    /// Stored as 4 components rather than 3: the xyz components are the tangent
    /// direction, and the w component is either +1 or -1. The w value is used
    /// to compute the bitangent (the third axis of tangent space) as:
    ///   bitangent = cross(normal, tangent.xyz) * tangent.w
    /// This is the convention used by GLTF.
    pub tangent: [f32; 4],
}

/// Describes a single drawable surface: a contiguous range of triangles in the
/// scene's shared index buffer, all sharing the same material.
///
/// Rather than each mesh owning its own vertex and index arrays, all meshes in
/// the scene share a single flat vertex buffer and a single flat index buffer.
/// A Mesh is just a window into those buffers — it says "my triangles are the
/// ones starting at `first_index` and running for `index_count` entries".
///
/// This layout is efficient for the GPU: the entire scene can be uploaded as
/// two buffers (vertices and indices) without any per-mesh overhead.
pub struct Mesh {
    /// The position in the scene's index buffer where this mesh's triangles begin.
    pub first_index: u32,

    /// How many index entries this mesh uses. Always a multiple of 3 because
    /// every three consecutive indices describe one triangle.
    pub index_count: u32,

    /// Which material to use when shading this mesh's surface. This is an index
    /// into Scene::materials.
    pub material_index: u32,
}

/// A specific occurrence of a mesh placed somewhere in the world.
///
/// The same mesh geometry can be reused in multiple places — imagine a scene
/// with 50 identical chairs. Instead of storing 50 copies of the chair's
/// vertices, we store the geometry once (as a Mesh) and create 50 MeshInstances,
/// each with a different transform saying where that chair is located, how it's
/// rotated, and how big it is.
///
/// This is called "instancing" and saves significant memory and load time for
/// scenes with repeated objects.
pub struct MeshInstance {
    /// Which mesh geometry to use, as an index into Scene::meshes.
    pub mesh_index: u32,

    /// A 4×4 matrix that transforms vertex positions from the mesh's local
    /// coordinate space into world space (the global coordinate system of
    /// the scene). This matrix encodes the object's position, rotation, and
    /// scale all in one.
    ///
    /// Stored in column-major order to match the GLTF file format and the
    /// convention used by the glam math library and WebGPU shaders.
    pub transform: [[f32; 4]; 4],
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    use std::mem::{offset_of, size_of};

    /// Changing field order or types in Vertex silently breaks the GPU shader.
    /// This test makes that mistake loud and immediate.
    #[test]
    fn test_vertex_gpu_layout() {
        assert_eq!(size_of::<Vertex>(),           48);
        assert_eq!(offset_of!(Vertex, position),   0);
        assert_eq!(offset_of!(Vertex, normal),    12);
        assert_eq!(offset_of!(Vertex, uv),        24);
        assert_eq!(offset_of!(Vertex, tangent),   32);
        let _: &[u8] = bytemuck::bytes_of(&Vertex::zeroed());
    }
}
