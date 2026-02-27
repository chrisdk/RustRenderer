pub struct Vertex {
    pub position: [f32; 3],
    pub normal:   [f32; 3],
    pub uv:       [f32; 2],
    pub tangent:  [f32; 4], // xyz + w for bitangent sign (GLTF convention)
}

pub struct Mesh {
    pub first_index:    u32, // offset into the scene index buffer
    pub index_count:    u32, // always a multiple of 3
    pub material_index: u32,
}

pub struct MeshInstance {
    pub mesh_index: u32,
    pub transform:  [[f32; 4]; 4], // column-major world-space transform
}
