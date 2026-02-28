//! Bounding Volume Hierarchy (BVH) acceleration structure.
//!
//! # What is a BVH?
//!
//! A BVH is a tree where every node has an **axis-aligned bounding box** (AABB)
//! — a box, aligned to the X/Y/Z axes, that tightly wraps all the geometry
//! below that node. Leaf nodes wrap a small number of triangles directly;
//! internal nodes wrap their two children.
//!
//! When a ray is shot into the scene, traversal works like this:
//! 1. Test the ray against the root node's bounding box. If it misses, the ray
//!    hits nothing — done in one test instead of millions.
//! 2. If it hits, recurse into the two children, testing each child's box.
//! 3. Skip any subtree whose box the ray misses.
//! 4. When a leaf is reached, test the ray against its actual triangles.
//!
//! In practice this reduces the number of triangle tests from O(n) to
//! O(log n), making real-time path tracing feasible.
//!
//! # Flat array layout
//!
//! For efficient GPU traversal the tree is stored as a flat `Vec<BvhNode>`,
//! not as a tree of pointers. This lets the GPU read nodes sequentially from
//! a buffer with no pointer chasing.
//!
//! Nodes are laid out in **pre-order** (a node always appears before its
//! children). For every internal node, the **left** child is always at index
//! `this_node_index + 1`, so it never needs to be stored. The **right** child
//! index is stored explicitly in [`BvhNode::right_or_first`].
//!
//! # Build algorithm
//!
//! The build uses a simple **median split** on the **longest axis**:
//! 1. Compute the bounding box of all triangle centroids in the current node.
//! 2. Find the axis (X, Y, or Z) along which the centroids are most spread out.
//! 3. Sort triangles by their centroid along that axis.
//! 4. Split at the median, so each half gets roughly the same number of
//!    triangles.
//! 5. Recurse into both halves.
//! 6. If the number of triangles is small enough, make a leaf instead.
//!
//! This is not the highest-quality strategy (SAH — Surface Area Heuristic —
//! gives better trees) but it is simple, fast to build, and good enough for
//! a first implementation.

use crate::scene::Scene;

// ============================================================================
// Constants
// ============================================================================

/// Maximum number of triangles a leaf node may contain.
///
/// Leaves with more triangles would be split further. Smaller values mean
/// deeper trees with fewer triangle tests per leaf; larger values mean
/// shallower trees with more tests per leaf. 4 is a common sweet spot.
const MAX_LEAF_TRIS: usize = 4;

// ============================================================================
// Public types
// ============================================================================

/// A single node in the flat BVH array.
///
/// Every node — whether a leaf or an internal node — carries an axis-aligned
/// bounding box that covers all geometry below it. The `count` field tells
/// the traversal code which kind of node this is and how to interpret
/// `right_or_first`.
///
/// This struct is exactly 32 bytes — a multiple of 16, satisfying WGSL's
/// storage-buffer array-element alignment requirement. `#[repr(C)]` and
/// `bytemuck::Pod` allow zero-copy upload: `bytemuck::cast_slice(&nodes)`
/// hands the GPU a raw byte slice with no intermediate allocation.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BvhNode {
    /// The minimum corner (smallest X, Y, Z) of this node's bounding box.
    pub aabb_min: [f32; 3],

    /// **Internal node** (`count == 0`): the index of the **right** child in
    /// [`Bvh::nodes`]. The left child is always at `this_node_index + 1`.
    ///
    /// **Leaf node** (`count > 0`): the index of the **first triangle** in
    /// [`Bvh::triangles`] covered by this leaf.
    pub right_or_first: u32,

    /// The maximum corner (largest X, Y, Z) of this node's bounding box.
    pub aabb_max: [f32; 3],

    /// `0` for internal nodes; the number of triangles (≥ 1) for leaf nodes.
    pub count: u32,
}

/// A triangle stored in world space, in the order the BVH was built.
///
/// After the BVH is built the triangles are reordered so that all triangles
/// belonging to a leaf are stored contiguously. This avoids random memory
/// access during traversal and is cache-friendly on the GPU.
///
/// Everything the shader needs to shade a hit point is packed here:
/// world-space positions for intersection, world-space normals for lighting,
/// UV coordinates for texture sampling, per-vertex tangents for normal mapping,
/// and the material index.
///
/// 160 bytes — a multiple of 16 as required for WGSL storage-buffer elements.
/// `#[repr(C)]` and `bytemuck::Pod` allow zero-copy GPU upload.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BvhTriangle {
    // ── World-space vertex positions (for ray–triangle intersection) ──────────

    /// World-space position of the first vertex.
    pub v0: [f32; 3],   // offset  0
    /// World-space position of the second vertex.
    pub v1: [f32; 3],   // offset 12
    /// World-space position of the third vertex.
    pub v2: [f32; 3],   // offset 24

    // ── Vertex indices and material ───────────────────────────────────────────

    /// Index of the first vertex in [`Scene::vertices`], for shading data.
    pub i0: u32,             // offset 36
    /// Index of the second vertex in [`Scene::vertices`], for shading data.
    pub i1: u32,             // offset 40
    /// Index of the third vertex in [`Scene::vertices`], for shading data.
    pub i2: u32,             // offset 44
    /// Index into [`Scene::materials`] for shading this triangle's surface.
    pub material_index: u32, // offset 48

    // ── World-space vertex normals (for smooth shading) ───────────────────────
    //
    // Baking these in at BVH build time (rather than looking them up via i0/i1/i2
    // at shader time) lets us apply the correct world-space transform once on the
    // CPU, keep the GPU shader simple, and avoid a separate vertex buffer binding.

    /// World-space normal at vertex 0.
    pub n0: [f32; 3],   // offset 52
    /// World-space normal at vertex 1.
    pub n1: [f32; 3],   // offset 64
    /// World-space normal at vertex 2.
    pub n2: [f32; 3],   // offset 76

    // ── UV texture coordinates ────────────────────────────────────────────────

    /// UV coordinates at vertex 0.
    pub uv0: [f32; 2],  // offset 88
    /// UV coordinates at vertex 1.
    pub uv1: [f32; 2],  // offset 96
    /// UV coordinates at vertex 2.
    pub uv2: [f32; 2],  // offset 104

    // ── World-space vertex tangents (for normal mapping) ─────────────────────
    //
    // GLTF normal maps are baked against Mikktspace tangents, stored per-vertex
    // in the file. Using these directly (rather than deriving tangents from
    // triangle edges) gives correct results at UV seams and mirrored UV islands.
    //
    // xyz = world-space tangent direction (normalised)
    // w   = handedness: +1 or -1. The bitangent is cross(N, T) * w, which
    //       corrects the sign for UV islands whose winding is mirrored.

    /// World-space tangent at vertex 0 (xyz = direction, w = handedness).
    pub t0: [f32; 4],   // offset 112
    /// World-space tangent at vertex 1.
    pub t1: [f32; 4],   // offset 128
    /// World-space tangent at vertex 2.
    pub t2: [f32; 4],   // offset 144
}

/// The fully built BVH acceleration structure.
///
/// Created once from a [`Scene`] by calling [`Bvh::build`] and then kept alive
/// for the entire rendering session. Both arrays are uploaded to the GPU as
/// storage buffers so the path tracing shader can traverse the tree.
pub struct Bvh {
    /// The flat node array. The root is always at index 0.
    pub nodes: Vec<BvhNode>,

    /// World-space triangles, reordered to match BVH leaf order.
    pub triangles: Vec<BvhTriangle>,
}

impl Bvh {
    /// Builds a BVH from all geometry in `scene`.
    ///
    /// All mesh instances are expanded into world-space triangles (applying
    /// each instance's transform), and the BVH is built over the combined set.
    /// After this call, [`Bvh::nodes`] and [`Bvh::triangles`] are ready to be
    /// uploaded to the GPU.
    pub fn build(scene: &Scene) -> Self {
        let mut tris = collect_triangles(scene);

        if tris.is_empty() {
            return Bvh { nodes: Vec::new(), triangles: Vec::new() };
        }

        let mut nodes = Vec::new();
        build_node(&mut nodes, &mut tris, 0);

        let triangles = tris
            .into_iter()
            .map(|t| BvhTriangle {
                v0: t.v0, v1: t.v1, v2: t.v2,
                i0: t.i0, i1: t.i1, i2: t.i2,
                material_index: t.material_index,
                n0: t.n0, n1: t.n1, n2: t.n2,
                uv0: t.uv0, uv1: t.uv1, uv2: t.uv2,
                t0: t.t0, t1: t.t1, t2: t.t2,
            })
            .collect();

        Bvh { nodes, triangles }
    }
}

// ============================================================================
// Private helpers
// ============================================================================

/// An axis-aligned bounding box, used internally during the build.
struct Aabb {
    min: [f32; 3],
    max: [f32; 3],
}

/// A triangle with all the data needed to build the BVH.
///
/// Stored separately from `BvhTriangle` because the centroid is only needed
/// during the build (for sorting) and not at traversal time.
struct BuildTriangle {
    v0:       [f32; 3],
    v1:       [f32; 3],
    v2:       [f32; 3],
    /// The average of the three vertex positions, used to decide which half of
    /// the split the triangle belongs to.
    centroid: [f32; 3],
    i0: u32, i1: u32, i2: u32,
    n0: [f32; 3], n1: [f32; 3], n2: [f32; 3],
    uv0: [f32; 2], uv1: [f32; 2], uv2: [f32; 2],
    /// World-space tangents (xyz = direction, w = handedness). The xyz
    /// component is rotated with the instance transform; w is unchanged.
    t0: [f32; 4], t1: [f32; 4], t2: [f32; 4],
    material_index: u32,
}

/// Iterates all mesh instances in the scene and emits one `BuildTriangle` per
/// triangle, with vertex positions and normals transformed into world space.
fn collect_triangles(scene: &Scene) -> Vec<BuildTriangle> {
    let mut result = Vec::new();

    for instance in &scene.instances {
        let mesh  = &scene.meshes[instance.mesh_index as usize];
        let xform = &instance.transform;

        let first = mesh.first_index as usize;
        let count = mesh.index_count as usize;

        // Three indices per triangle; step by 3.
        for tri_base in (first..first + count).step_by(3) {
            let i0 = scene.indices[tri_base];
            let i1 = scene.indices[tri_base + 1];
            let i2 = scene.indices[tri_base + 2];

            let vert0 = &scene.vertices[i0 as usize];
            let vert1 = &scene.vertices[i1 as usize];
            let vert2 = &scene.vertices[i2 as usize];

            let v0 = transform_point(xform, vert0.position);
            let v1 = transform_point(xform, vert1.position);
            let v2 = transform_point(xform, vert2.position);

            // Transform normals with the upper-left 3×3 of the matrix and
            // renormalise. This is exact for rotation/translation/uniform-scale
            // transforms (which cover the vast majority of real scenes). For
            // non-uniform scale the proper transform is the inverse transpose,
            // but renormalising is a good-enough approximation.
            let n0 = transform_normal(xform, vert0.normal);
            let n1 = transform_normal(xform, vert1.normal);
            let n2 = transform_normal(xform, vert2.normal);

            // Tangents transform the same way as normals (direction vectors),
            // except the w handedness component is a scalar — it survives the
            // transform unchanged. The sign encodes whether the UV island is
            // mirrored; no rotation can flip that.
            let t0 = transform_tangent(xform, vert0.tangent);
            let t1 = transform_tangent(xform, vert1.tangent);
            let t2 = transform_tangent(xform, vert2.tangent);

            let centroid = [
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
                (v0[2] + v1[2] + v2[2]) / 3.0,
            ];

            result.push(BuildTriangle {
                v0, v1, v2, centroid,
                i0, i1, i2,
                n0, n1, n2,
                uv0: vert0.uv, uv1: vert1.uv, uv2: vert2.uv,
                t0, t1, t2,
                material_index: mesh.material_index,
            });
        }
    }

    result
}

/// Recursively builds BVH nodes for the triangles in `tris`, appending new
/// nodes to `nodes` and returning the index of the node just created.
///
/// `global_start` is the position of `tris[0]` in the global triangle array,
/// used to fill in the `right_or_first` field of leaf nodes.
///
/// Nodes are written in **pre-order**: this node is pushed first, so its left
/// child is always at `this_node_index + 1` — no need to store it explicitly.
fn build_node(
    nodes: &mut Vec<BvhNode>,
    tris:  &mut [BuildTriangle],
    global_start: usize,
) -> usize {
    // Reserve a slot for this node. Children will be pushed after this.
    let node_idx = nodes.len();
    nodes.push(BvhNode { aabb_min: [0.0; 3], right_or_first: 0, aabb_max: [0.0; 3], count: 0 });

    let aabb = compute_aabb(tris);

    // Base case: small enough to be a leaf.
    if tris.len() <= MAX_LEAF_TRIS {
        nodes[node_idx] = BvhNode {
            aabb_min:       aabb.min,
            right_or_first: global_start as u32,
            aabb_max:       aabb.max,
            count:          tris.len() as u32,
        };
        return node_idx;
    }

    // Sort triangles by their centroid along the longest axis of the AABB.
    let axis = longest_axis(&aabb);
    tris.sort_unstable_by(|a, b| a.centroid[axis].total_cmp(&b.centroid[axis]));

    let mid = tris.len() / 2;

    // Build the left child. Because we already pushed our placeholder at
    // node_idx, the left child will be pushed at node_idx + 1.
    build_node(nodes, &mut tris[..mid], global_start);

    // Build the right child. It follows the entire left subtree.
    let right_idx = build_node(nodes, &mut tris[mid..], global_start + mid);

    nodes[node_idx] = BvhNode {
        aabb_min:       aabb.min,
        right_or_first: right_idx as u32,
        aabb_max:       aabb.max,
        count:          0, // 0 signals an internal node
    };

    node_idx
}

/// Computes the tightest axis-aligned bounding box that contains all vertices
/// of every triangle in `tris`.
fn compute_aabb(tris: &[BuildTriangle]) -> Aabb {
    let mut min = [f32::INFINITY;     3];
    let mut max = [f32::NEG_INFINITY; 3];

    for tri in tris {
        for v in [tri.v0, tri.v1, tri.v2] {
            for i in 0..3 {
                if v[i] < min[i] { min[i] = v[i]; }
                if v[i] > max[i] { max[i] = v[i]; }
            }
        }
    }

    Aabb { min, max }
}

/// Returns the index of the axis (0=X, 1=Y, 2=Z) along which `aabb` is widest.
///
/// Splitting along the widest axis tends to produce the most balanced BVH.
fn longest_axis(aabb: &Aabb) -> usize {
    let dx = aabb.max[0] - aabb.min[0];
    let dy = aabb.max[1] - aabb.min[1];
    let dz = aabb.max[2] - aabb.min[2];
    if dx >= dy && dx >= dz { 0 }
    else if dy >= dz         { 1 }
    else                      { 2 }
}

/// Multiplies a 3D point by a column-major 4×4 homogeneous transform matrix.
///
/// The matrix is stored as `m[column][row]`, which is the layout used by GLTF
/// and `glam`. The w component of the result is discarded — we only need the
/// transformed (x, y, z) position, not the homogeneous w.
fn transform_point(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0]*p[0] + m[1][0]*p[1] + m[2][0]*p[2] + m[3][0],
        m[0][1]*p[0] + m[1][1]*p[1] + m[2][1]*p[2] + m[3][1],
        m[0][2]*p[0] + m[1][2]*p[1] + m[2][2]*p[2] + m[3][2],
    ]
}

/// Transforms a surface normal by the upper-left 3×3 of a column-major 4×4
/// matrix, then renormalises the result.
///
/// Normals are direction vectors, not positions, so the translation column is
/// ignored. Strictly correct normal transformation requires the inverse
/// transpose of the matrix (to preserve perpendicularity under non-uniform
/// scale), but multiplying by the original 3×3 and renormalising is accurate
/// for rotation, translation, and uniform scale — the common cases in practice.
fn transform_normal(m: &[[f32; 4]; 4], n: [f32; 3]) -> [f32; 3] {
    let x = m[0][0]*n[0] + m[1][0]*n[1] + m[2][0]*n[2];
    let y = m[0][1]*n[0] + m[1][1]*n[1] + m[2][1]*n[2];
    let z = m[0][2]*n[0] + m[1][2]*n[1] + m[2][2]*n[2];
    let len = (x*x + y*y + z*z).sqrt();
    if len > 1e-8 { [x/len, y/len, z/len] } else { n }
}

/// Transforms a GLTF tangent vector `[tx, ty, tz, w]` by a column-major 4×4
/// matrix.
///
/// The xyz component is rotated exactly like a normal (upper-left 3×3, then
/// renormalise). The w component — the handedness bit (+1 or −1) that tells
/// the shader which way the bitangent points — is a pure scalar and passes
/// through unchanged: no spatial rotation can flip the winding of a UV island.
fn transform_tangent(m: &[[f32; 4]; 4], t: [f32; 4]) -> [f32; 4] {
    let xyz = transform_normal(m, [t[0], t[1], t[2]]);
    [xyz[0], xyz[1], xyz[2], t[3]]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    use crate::scene::{
        Scene,
        geometry::{Vertex, Mesh, MeshInstance},
        material::Material,
    };

    // -------------------------------------------------------------------------
    // Test helper
    // -------------------------------------------------------------------------

    /// Builds a minimal [`Scene`] from a slice of world-space triangles, each
    /// given as three vertex positions. All triangles share one mesh and one
    /// instance with the identity transform.
    fn scene_from_triangles(tris: &[([f32; 3], [f32; 3], [f32; 3])]) -> Scene {
        let mut vertices = Vec::new();
        let mut indices  = Vec::new();

        for &(p0, p1, p2) in tris {
            let base = vertices.len() as u32;
            let make = |p: [f32; 3]| Vertex {
                position: p,
                normal:   [0.0, 1.0, 0.0],
                uv:       [0.0, 0.0],
                tangent:  [1.0, 0.0, 0.0, 1.0],
            };
            vertices.push(make(p0));
            vertices.push(make(p1));
            vertices.push(make(p2));
            indices.extend_from_slice(&[base, base + 1, base + 2]);
        }

        let mesh = Mesh {
            first_index:    0,
            index_count:    (tris.len() * 3) as u32,
            material_index: 0,
        };
        // Identity transform: column-major 4×4 identity matrix.
        let identity = [
            [1.0_f32, 0.0, 0.0, 0.0],
            [0.0_f32, 1.0, 0.0, 0.0],
            [0.0_f32, 0.0, 1.0, 0.0],
            [0.0_f32, 0.0, 0.0, 1.0],
        ];
        let instance = MeshInstance { mesh_index: 0, transform: identity };

        Scene {
            vertices,
            indices,
            meshes:      vec![mesh],
            instances:   vec![instance],
            materials:   vec![Material::default()],
            textures:    Vec::new(),
            environment: None,
        }
    }

    // -------------------------------------------------------------------------
    // Tests
    // -------------------------------------------------------------------------

    /// Guard the GPU buffer layout: if someone changes a field type or order,
    /// the WGSL struct definitions will silently misread memory on the GPU.
    /// Better to catch it here as a compile-time-like assertion.
    #[test]
    fn test_bvh_node_gpu_layout() {
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<BvhNode>(),                   32);
        assert_eq!(offset_of!(BvhNode, aabb_min),           0);
        assert_eq!(offset_of!(BvhNode, right_or_first),    12);
        assert_eq!(offset_of!(BvhNode, aabb_max),          16);
        assert_eq!(offset_of!(BvhNode, count),             28);
        // Verify bytemuck can actually cast it (would panic if not Pod)
        let _: &[u8] = bytemuck::bytes_of(&BvhNode::zeroed());
    }

    #[test]
    fn test_bvh_triangle_gpu_layout() {
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<BvhTriangle>(),                   160);
        assert_eq!(offset_of!(BvhTriangle, v0),                  0);
        assert_eq!(offset_of!(BvhTriangle, v1),                 12);
        assert_eq!(offset_of!(BvhTriangle, v2),                 24);
        assert_eq!(offset_of!(BvhTriangle, i0),                 36);
        assert_eq!(offset_of!(BvhTriangle, i1),                 40);
        assert_eq!(offset_of!(BvhTriangle, i2),                 44);
        assert_eq!(offset_of!(BvhTriangle, material_index),     48);
        assert_eq!(offset_of!(BvhTriangle, n0),                 52);
        assert_eq!(offset_of!(BvhTriangle, n1),                 64);
        assert_eq!(offset_of!(BvhTriangle, n2),                 76);
        assert_eq!(offset_of!(BvhTriangle, uv0),                88);
        assert_eq!(offset_of!(BvhTriangle, uv1),                96);
        assert_eq!(offset_of!(BvhTriangle, uv2),               104);
        assert_eq!(offset_of!(BvhTriangle, t0),                112);
        assert_eq!(offset_of!(BvhTriangle, t1),                128);
        assert_eq!(offset_of!(BvhTriangle, t2),                144);
        let _: &[u8] = bytemuck::bytes_of(&BvhTriangle::zeroed());
    }

    /// An empty scene should produce an empty BVH without panicking.
    #[test]
    fn test_empty_scene() {
        let scene = Scene {
            vertices:    Vec::new(),
            indices:     Vec::new(),
            meshes:      Vec::new(),
            instances:   Vec::new(),
            materials:   Vec::new(),
            textures:    Vec::new(),
            environment: None,
        };
        let bvh = Bvh::build(&scene);
        assert!(bvh.nodes.is_empty());
        assert!(bvh.triangles.is_empty());
    }

    /// A scene with a single triangle should produce exactly one leaf node.
    #[test]
    fn test_single_triangle() {
        let scene = scene_from_triangles(&[
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ]);
        let bvh = Bvh::build(&scene);

        assert_eq!(bvh.triangles.len(), 1, "should have 1 triangle");
        assert_eq!(bvh.nodes.len(),     1, "should have 1 node (the root leaf)");

        let root = &bvh.nodes[0];
        assert_eq!(root.count, 1, "root should be a leaf with 1 triangle");
        assert_eq!(root.right_or_first, 0, "leaf starts at triangle index 0");
    }

    /// The root node's AABB must contain every vertex of every triangle.
    #[test]
    fn test_aabb_contains_all_vertices() {
        let scene = scene_from_triangles(&[
            ([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),
        ]);
        let bvh   = Bvh::build(&scene);
        let root  = &bvh.nodes[0];

        assert!(root.aabb_min[0] <= -1.0, "min x should be ≤ -1");
        assert!(root.aabb_min[1] <= -2.0, "min y should be ≤ -2");
        assert!(root.aabb_min[2] <= -3.0, "min z should be ≤ -3");
        assert!(root.aabb_max[0] >=  1.0, "max x should be ≥ 1");
        assert!(root.aabb_max[1] >=  2.0, "max y should be ≥ 2");
        assert!(root.aabb_max[2] >=  3.0, "max z should be ≥ 3");
    }

    /// Every triangle must appear in exactly one leaf node — no gaps, no
    /// duplicates. This verifies that the build doesn't lose or double-count
    /// any geometry.
    #[test]
    fn test_leaf_coverage() {
        // 5 triangles spread along the X axis forces at least one internal node
        // (MAX_LEAF_TRIS = 4).
        let scene = scene_from_triangles(&[
            ([ 0.0, 0.0, 0.0], [ 1.0, 0.0, 0.0], [ 0.0, 1.0, 0.0]),
            ([ 2.0, 0.0, 0.0], [ 3.0, 0.0, 0.0], [ 2.0, 1.0, 0.0]),
            ([ 4.0, 0.0, 0.0], [ 5.0, 0.0, 0.0], [ 4.0, 1.0, 0.0]),
            ([ 6.0, 0.0, 0.0], [ 7.0, 0.0, 0.0], [ 6.0, 1.0, 0.0]),
            ([ 8.0, 0.0, 0.0], [ 9.0, 0.0, 0.0], [ 8.0, 1.0, 0.0]),
        ]);
        let bvh = Bvh::build(&scene);

        let total = bvh.triangles.len();
        let mut covered = vec![false; total];

        for node in &bvh.nodes {
            if node.count > 0 {
                // Leaf: mark its triangle range as covered.
                for i in 0..node.count as usize {
                    let idx = node.right_or_first as usize + i;
                    assert!(idx < total,     "leaf triangle index out of bounds");
                    assert!(!covered[idx],   "triangle {idx} is covered by two leaves");
                    covered[idx] = true;
                }
            }
        }

        assert!(covered.iter().all(|&c| c), "some triangles are not covered by any leaf");
    }

    /// Every internal node's AABB must fully contain the AABBs of both its
    /// children. This confirms that the tree correctly encloses all geometry.
    #[test]
    fn test_parent_contains_children() {
        let scene = scene_from_triangles(&[
            ([ 0.0, 0.0, 0.0], [ 1.0, 0.0, 0.0], [ 0.0, 1.0, 0.0]),
            ([ 5.0, 0.0, 0.0], [ 6.0, 0.0, 0.0], [ 5.0, 1.0, 0.0]),
            ([10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0]),
            ([15.0, 0.0, 0.0], [16.0, 0.0, 0.0], [15.0, 1.0, 0.0]),
            ([20.0, 0.0, 0.0], [21.0, 0.0, 0.0], [20.0, 1.0, 0.0]),
        ]);
        let bvh = Bvh::build(&scene);

        for (idx, node) in bvh.nodes.iter().enumerate() {
            if node.count != 0 {
                continue; // skip leaves
            }
            // Internal node: left child is at idx+1, right child at right_or_first.
            let left  = &bvh.nodes[idx + 1];
            let right = &bvh.nodes[node.right_or_first as usize];

            for axis in 0..3 {
                assert!(
                    node.aabb_min[axis] <= left.aabb_min[axis],
                    "node {idx} min[{axis}] ({}) > left child min ({})",
                    node.aabb_min[axis], left.aabb_min[axis],
                );
                assert!(
                    node.aabb_max[axis] >= left.aabb_max[axis],
                    "node {idx} max[{axis}] ({}) < left child max ({})",
                    node.aabb_max[axis], left.aabb_max[axis],
                );
                assert!(
                    node.aabb_min[axis] <= right.aabb_min[axis],
                    "node {idx} min[{axis}] ({}) > right child min ({})",
                    node.aabb_min[axis], right.aabb_min[axis],
                );
                assert!(
                    node.aabb_max[axis] >= right.aabb_max[axis],
                    "node {idx} max[{axis}] ({}) < right child max ({})",
                    node.aabb_max[axis], right.aabb_max[axis],
                );
            }
        }
    }

    /// A mesh instance with a translation transform should produce world-space
    /// triangle vertices at the translated positions.
    #[test]
    fn test_world_transform_applied() {
        // Translate the mesh by (10, 0, 0) using a column-major matrix.
        let translate_x10 = [
            [1.0_f32, 0.0, 0.0, 0.0], // col 0
            [0.0_f32, 1.0, 0.0, 0.0], // col 1
            [0.0_f32, 0.0, 1.0, 0.0], // col 2
            [10.0_f32, 0.0, 0.0, 1.0], // col 3: translation
        ];

        let vertices = vec![
            Vertex { position: [0.0, 0.0, 0.0], normal: [0.0,1.0,0.0], uv:[0.0,0.0], tangent:[1.0,0.0,0.0,1.0] },
            Vertex { position: [1.0, 0.0, 0.0], normal: [0.0,1.0,0.0], uv:[1.0,0.0], tangent:[1.0,0.0,0.0,1.0] },
            Vertex { position: [0.0, 1.0, 0.0], normal: [0.0,1.0,0.0], uv:[0.0,1.0], tangent:[1.0,0.0,0.0,1.0] },
        ];
        let scene = Scene {
            indices:   vec![0, 1, 2],
            meshes:    vec![Mesh { first_index: 0, index_count: 3, material_index: 0 }],
            instances: vec![MeshInstance { mesh_index: 0, transform: translate_x10 }],
            materials: vec![Material::default()],
            vertices,
            textures:    Vec::new(),
            environment: None,
        };

        let bvh = Bvh::build(&scene);
        assert_eq!(bvh.triangles.len(), 1);

        let tri = &bvh.triangles[0];
        // v0 was [0,0,0] → world [10,0,0]
        assert!((tri.v0[0] - 10.0).abs() < 1e-5, "v0.x = {}, want 10", tri.v0[0]);
        assert!(tri.v0[1].abs() < 1e-5);
        assert!(tri.v0[2].abs() < 1e-5);
        // v1 was [1,0,0] → world [11,0,0]
        assert!((tri.v1[0] - 11.0).abs() < 1e-5, "v1.x = {}, want 11", tri.v1[0]);
    }

    /// Pre-order invariant: for every internal node at index `i`, the left child
    /// is at `i + 1`. This is load-bearing — the traversal code (and the GPU
    /// shader) relies on it and never stores the left child index explicitly.
    #[test]
    fn test_left_child_always_at_parent_plus_one() {
        // 5 triangles forces at least one split (MAX_LEAF_TRIS = 4).
        let scene = scene_from_triangles(&[
            ([ 0., 0., 0.], [ 1., 0., 0.], [ 0., 1., 0.]),
            ([ 3., 0., 0.], [ 4., 0., 0.], [ 3., 1., 0.]),
            ([ 6., 0., 0.], [ 7., 0., 0.], [ 6., 1., 0.]),
            ([ 9., 0., 0.], [10., 0., 0.], [ 9., 1., 0.]),
            ([12., 0., 0.], [13., 0., 0.], [12., 1., 0.]),
        ]);
        let bvh = Bvh::build(&scene);

        for (idx, node) in bvh.nodes.iter().enumerate() {
            if node.count != 0 { continue; } // leaf — no children to check

            // Left child must be the very next node in the array.
            assert!(idx + 1 < bvh.nodes.len(),
                "internal node {idx}: no room for left child at idx+1");

            // Right child index must be in-bounds and strictly after the left child.
            let right = node.right_or_first as usize;
            assert!(right < bvh.nodes.len(),
                "internal node {idx}: right_or_first ({right}) out of bounds");
            assert!(right > idx + 1,
                "internal node {idx}: right child ({right}) must follow left child ({})",
                idx + 1);
        }
    }

    /// With exactly MAX_LEAF_TRIS triangles the root should be one leaf.
    /// One more triangle must trigger a split into at least two nodes.
    /// This pins the split-threshold boundary so refactoring can't silently
    /// change when leaves are created.
    #[test]
    fn test_split_at_max_leaf_tris_boundary() {
        let make_scene = |n: usize| {
            let tris: Vec<_> = (0..n).map(|i| {
                let x = i as f32 * 2.0;
                ([x, 0., 0.], [x+1., 0., 0.], [x, 1., 0.])
            }).collect();
            scene_from_triangles(&tris)
        };

        // Exactly MAX_LEAF_TRIS — should be a single leaf.
        let bvh4 = Bvh::build(&make_scene(MAX_LEAF_TRIS));
        assert_eq!(bvh4.nodes.len(), 1,
            "{MAX_LEAF_TRIS} triangles should produce a single root leaf");
        assert_eq!(bvh4.nodes[0].count as usize, MAX_LEAF_TRIS,
            "root leaf should hold all {MAX_LEAF_TRIS} triangles");

        // One over — must split.
        let bvh5 = Bvh::build(&make_scene(MAX_LEAF_TRIS + 1));
        assert!(bvh5.nodes.len() > 1,
            "{} triangles must produce at least one internal node", MAX_LEAF_TRIS + 1);
        assert_eq!(bvh5.nodes[0].count, 0,
            "root of a split tree should be an internal node (count=0)");
    }

    /// Triangle reordering during BVH construction must preserve each triangle's
    /// material_index. If this breaks, triangles render with the wrong material.
    #[test]
    fn test_material_index_preserved() {
        // Two meshes with different material indices, placed far apart so the
        // BVH will put them in separate leaves after splitting.
        let vertices = vec![
            Vertex { position: [ 0., 0., 0.], normal:[0.,1.,0.], uv:[0.,0.], tangent:[1.,0.,0.,1.] },
            Vertex { position: [ 1., 0., 0.], normal:[0.,1.,0.], uv:[1.,0.], tangent:[1.,0.,0.,1.] },
            Vertex { position: [ 0., 1., 0.], normal:[0.,1.,0.], uv:[0.,1.], tangent:[1.,0.,0.,1.] },
            Vertex { position: [10., 0., 0.], normal:[0.,1.,0.], uv:[0.,0.], tangent:[1.,0.,0.,1.] },
            Vertex { position: [11., 0., 0.], normal:[0.,1.,0.], uv:[1.,0.], tangent:[1.,0.,0.,1.] },
            Vertex { position: [10., 1., 0.], normal:[0.,1.,0.], uv:[0.,1.], tangent:[1.,0.,0.,1.] },
        ];
        let identity = [
            [1.0_f32, 0., 0., 0.], [0.0_f32, 1., 0., 0.],
            [0.0_f32, 0., 1., 0.], [0.0_f32, 0., 0., 1.],
        ];
        let scene = Scene {
            vertices,
            indices:   vec![0, 1, 2,  3, 4, 5],
            meshes:    vec![
                Mesh { first_index: 0, index_count: 3, material_index: 0 },
                Mesh { first_index: 3, index_count: 3, material_index: 7 }, // non-zero material
            ],
            instances: vec![
                MeshInstance { mesh_index: 0, transform: identity },
                MeshInstance { mesh_index: 1, transform: identity },
            ],
            materials:   vec![Material::default(); 8],
            textures:    Vec::new(),
            environment: None,
        };

        let bvh = Bvh::build(&scene);
        assert_eq!(bvh.triangles.len(), 2);

        for tri in &bvh.triangles {
            let cx = (tri.v0[0] + tri.v1[0] + tri.v2[0]) / 3.0;
            let expected_mat = if cx < 5.0 { 0 } else { 7 };
            assert_eq!(tri.material_index, expected_mat,
                "triangle at centroid x≈{cx:.1} should have material {expected_mat}, \
                 got {}", tri.material_index);
        }
    }

    /// Two instances of the same mesh should each contribute their own triangle
    /// to the BVH. This verifies that instancing is expanded correctly.
    #[test]
    fn test_multiple_instances() {
        let vertices = vec![
            Vertex { position: [0.0, 0.0, 0.0], normal: [0.0,1.0,0.0], uv:[0.0,0.0], tangent:[1.0,0.0,0.0,1.0] },
            Vertex { position: [1.0, 0.0, 0.0], normal: [0.0,1.0,0.0], uv:[1.0,0.0], tangent:[1.0,0.0,0.0,1.0] },
            Vertex { position: [0.0, 1.0, 0.0], normal: [0.0,1.0,0.0], uv:[0.0,1.0], tangent:[1.0,0.0,0.0,1.0] },
        ];
        let identity = [
            [1.0_f32, 0.0, 0.0, 0.0],
            [0.0_f32, 1.0, 0.0, 0.0],
            [0.0_f32, 0.0, 1.0, 0.0],
            [0.0_f32, 0.0, 0.0, 1.0],
        ];
        let translate5 = [
            [1.0_f32, 0.0, 0.0, 0.0],
            [0.0_f32, 1.0, 0.0, 0.0],
            [0.0_f32, 0.0, 1.0, 0.0],
            [5.0_f32, 0.0, 0.0, 1.0],
        ];
        let scene = Scene {
            vertices,
            indices:   vec![0, 1, 2],
            meshes:    vec![Mesh { first_index: 0, index_count: 3, material_index: 0 }],
            instances: vec![
                MeshInstance { mesh_index: 0, transform: identity },
                MeshInstance { mesh_index: 0, transform: translate5 },
            ],
            materials:   vec![Material::default()],
            textures:    Vec::new(),
            environment: None,
        };

        let bvh = Bvh::build(&scene);
        // One mesh, two instances → two world-space triangles in the BVH.
        assert_eq!(bvh.triangles.len(), 2, "expected 2 triangles (one per instance)");
    }
}
