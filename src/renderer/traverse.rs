//! CPU BVH traversal — the ground truth for the WGSL shader.
//!
//! This is the most important module in Phase 1. The algorithm here is
//! deliberately written to mirror the WGSL shader as closely as Rust syntax
//! allows: same stack discipline, same child-ordering logic, same intersection
//! calls. When the GPU shader produces a wrong result, reproduce the failing
//! ray here first — you'll have a debugger, `println!`, and the unit tests
//! below to work with.
//!
//! # Algorithm overview
//!
//! Iterative depth-first traversal with an explicit stack (recursion doesn't
//! exist in WGSL). Start at the root (index 0). For each node:
//!
//! - **Leaf** (`count > 0`): test all its triangles with Möller–Trumbore.
//!   Update `t_max` on every hit so we only keep the closest one.
//! - **Internal** (`count == 0`): test both children's AABBs. Push the
//!   farther child first, nearer child on top. This "front-to-back" order
//!   lets us prune the far child sooner once we've found a closer hit.
//!
//! # BVH node layout reminder
//!
//! For an internal node at index `i`:
//! - Left child  = `nodes[i + 1]`   (implicit — never stored)
//! - Right child = `nodes[node.right_or_first as usize]`  (stored explicitly)

use crate::accel::bvh::Bvh;
use crate::camera::Ray;
use super::intersect::{ray_aabb, ray_triangle};

/// The maximum BVH stack depth. A tree balanced over a million triangles with
/// MAX_LEAF_TRIS = 4 is at most ~18 levels deep; 64 gives plenty of headroom
/// for unbalanced scenes while still fitting comfortably on the stack.
const STACK_SIZE: usize = 64;

// ============================================================================
// Public types
// ============================================================================

/// Everything the shading code needs to know about a ray–scene intersection.
///
/// After traversal you have a `triangle_index` into [`Bvh::triangles`] and
/// barycentric coordinates `(u, v)`. Use those to interpolate the shading
/// normal, UV coordinates, and tangent at the hit point.
#[derive(Debug, Clone, Copy)]
pub struct HitRecord {
    /// Distance along the ray to the hit point: `p = ray.origin + t * ray.direction`.
    pub t: f32,

    /// First barycentric coordinate. The hit point is:
    /// `p = (1 - u - v) * tri.v0 + u * tri.v1 + v * tri.v2`
    pub u: f32,

    /// Second barycentric coordinate (see `u`).
    pub v: f32,

    /// Index of the hit triangle in [`Bvh::triangles`].
    pub triangle_index: u32,
}

// ============================================================================
// Traversal
// ============================================================================

/// Finds the closest intersection of `ray` with any triangle in `bvh`.
///
/// Returns `None` if the ray misses everything. The caller typically passes
/// `t_min = 0.001` (a small epsilon to avoid self-intersection after a bounce)
/// and `t_max = f32::MAX` for primary rays.
///
/// This implementation is intentionally structured to match the WGSL compute
/// shader in `src/renderer/shaders/trace.wgsl`. Change one, change the other.
pub fn intersect_bvh(bvh: &Bvh, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
    if bvh.nodes.is_empty() {
        return None;
    }

    // Precompute reciprocal direction once. The slab test multiplies by this
    // instead of dividing by direction — three fewer divides per AABB test.
    let inv_dir = [
        1.0 / ray.direction[0],
        1.0 / ray.direction[1],
        1.0 / ray.direction[2],
    ];

    // Explicit stack. In WGSL this will be `var<private> stack: array<u32, 64>`.
    let mut stack     = [0u32; STACK_SIZE];
    let mut stack_top = 0i32;
    stack[0] = 0; // push root

    let mut closest = t_max;
    let mut hit: Option<HitRecord> = None;

    while stack_top >= 0 {
        let node_idx = stack[stack_top as usize] as usize;
        stack_top -= 1;

        let node = &bvh.nodes[node_idx];

        // Quick rejection: does this ray even enter the node's bounding box?
        if ray_aabb(ray.origin, inv_dir, t_min, closest,
                    node.aabb_min, node.aabb_max).is_none() {
            continue;
        }

        if node.count > 0 {
            // Leaf: test every triangle.
            let first = node.right_or_first as usize;
            for tri_idx in first..first + node.count as usize {
                let tri = &bvh.triangles[tri_idx];
                if let Some((t, u, v)) = ray_triangle(
                    ray.origin, ray.direction,
                    tri.v0, tri.v1, tri.v2,
                    t_min, closest,
                ) {
                    // Closer hit — shrink the search window and record it.
                    closest = t;
                    hit = Some(HitRecord { t, u, v, triangle_index: tri_idx as u32 });
                }
            }
        } else {
            // Internal node: test both children and push in front-to-back order.
            // The left child is at node_idx + 1 (by the pre-order build convention).
            let left_idx  = node_idx + 1;
            let right_idx = node.right_or_first as usize;

            let t_left  = ray_aabb(ray.origin, inv_dir, t_min, closest,
                                   bvh.nodes[left_idx].aabb_min,  bvh.nodes[left_idx].aabb_max);
            let t_right = ray_aabb(ray.origin, inv_dir, t_min, closest,
                                   bvh.nodes[right_idx].aabb_min, bvh.nodes[right_idx].aabb_max);

            match (t_left, t_right) {
                (None, None)         => { /* both children miss — skip */ }
                (Some(_), None)      => { push(&mut stack, &mut stack_top, left_idx as u32); }
                (None, Some(_))      => { push(&mut stack, &mut stack_top, right_idx as u32); }
                (Some(tl), Some(tr)) => {
                    // Push the farther child first so the nearer child ends up
                    // on top of the stack and is processed next.
                    if tl <= tr {
                        push(&mut stack, &mut stack_top, right_idx as u32);
                        push(&mut stack, &mut stack_top, left_idx as u32);
                    } else {
                        push(&mut stack, &mut stack_top, left_idx as u32);
                        push(&mut stack, &mut stack_top, right_idx as u32);
                    }
                }
            }
        }
    }

    hit
}

#[inline]
fn push(stack: &mut [u32; STACK_SIZE], top: &mut i32, idx: u32) {
    *top += 1;
    stack[*top as usize] = idx;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accel::bvh::Bvh;
    use crate::camera::Ray;
    use crate::scene::{
        Scene,
        geometry::{Vertex, Mesh, MeshInstance},
        material::Material,
    };

    const EPS: f32 = 1e-4;
    fn approx_eq(a: f32, b: f32) -> bool { (a - b).abs() < EPS }

    // ---- helpers ----

    fn make_vertex(p: [f32; 3]) -> Vertex {
        Vertex { position: p, normal: [0.,1.,0.], uv: [0.,0.], tangent: [1.,0.,0.,1.] }
    }

    fn identity_xform() -> [[f32; 4]; 4] {
        [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
    }

    /// Build a BVH from a list of world-space triangles (with an identity
    /// instance transform — positions are already in world space).
    fn bvh_from_tris(tris: &[([f32;3],[f32;3],[f32;3])]) -> Bvh {
        let mut vertices = Vec::new();
        let mut indices  = Vec::new();
        for &(a, b, c) in tris {
            let base = vertices.len() as u32;
            vertices.push(make_vertex(a));
            vertices.push(make_vertex(b));
            vertices.push(make_vertex(c));
            indices.extend_from_slice(&[base, base+1, base+2]);
        }
        let mesh = Mesh { first_index: 0, index_count: (tris.len()*3) as u32, material_index: 0 };
        let scene = Scene {
            vertices,
            indices,
            meshes:      vec![mesh],
            instances:   vec![MeshInstance { mesh_index: 0, transform: identity_xform() }],
            materials:   vec![Material::default()],
            textures:    Vec::new(),
            environment: None,
        };
        Bvh::build(&scene)
    }

    fn ray(ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32) -> Ray {
        let len = (dx*dx+dy*dy+dz*dz).sqrt();
        Ray { origin: [ox,oy,oz], direction: [dx/len,dy/len,dz/len] }
    }

    // ---- tests ----

    /// An empty BVH (no geometry) must return None without panicking.
    #[test]
    fn test_empty_bvh_returns_none() {
        let bvh = Bvh { nodes: vec![], triangles: vec![] };
        let r   = ray(0.,0.,0., 0.,0.,-1.);
        assert!(intersect_bvh(&bvh, &r, 0.001, 1e30).is_none());
    }

    /// A single triangle hit should produce a HitRecord with the correct t.
    #[test]
    fn test_single_triangle_hit() {
        let bvh = bvh_from_tris(&[
            ([-1.,-1.,-3.], [1.,-1.,-3.], [0.,1.,-3.]),
        ]);
        let r   = ray(0., 0., 0.,  0., 0., -1.);
        let hit = intersect_bvh(&bvh, &r, 0.001, 1e30);
        assert!(hit.is_some(), "should hit the triangle");
        assert!(approx_eq(hit.unwrap().t, 3.0), "t should be 3.0, got {}", hit.unwrap().t);
    }

    /// A ray aimed away from the triangle must return None.
    #[test]
    fn test_single_triangle_miss() {
        let bvh = bvh_from_tris(&[
            ([-1.,-1.,-3.], [1.,-1.,-3.], [0.,1.,-3.]),
        ]);
        let r = ray(0., 5., 0.,  0., 0., -1.); // ray displaced above triangle
        assert!(intersect_bvh(&bvh, &r, 0.001, 1e30).is_none());
    }

    /// Two triangles at different depths — we must get the nearer one.
    #[test]
    fn test_nearest_triangle_returned() {
        // Triangle A at z=-1, triangle B at z=-5
        let bvh = bvh_from_tris(&[
            ([-1.,-1.,-5.], [1.,-1.,-5.], [0.,1.,-5.]),  // index 0, far
            ([-1.,-1.,-1.], [1.,-1.,-1.], [0.,1.,-1.]),  // index 1, near
        ]);
        let r   = ray(0., 0., 0.,  0., 0., -1.);
        let hit = intersect_bvh(&bvh, &r, 0.001, 1e30).expect("should hit something");
        assert!(approx_eq(hit.t, 1.0), "should hit the nearer triangle at t=1, got {}", hit.t);
    }

    /// A ray aimed into empty space should miss even a populated scene.
    #[test]
    fn test_ray_misses_all_triangles() {
        let bvh = bvh_from_tris(&[
            ([ 0.,-1.,-2.], [2.,-1.,-2.], [1.,1.,-2.]),
            ([ 3.,-1.,-2.], [5.,-1.,-2.], [4.,1.,-2.]),
        ]);
        let r = ray(-10., 0., 0.,  0., 0., -1.); // way to the left of everything
        assert!(intersect_bvh(&bvh, &r, 0.001, 1e30).is_none());
    }

    /// The returned triangle_index must identify the actual triangle we hit.
    #[test]
    fn test_correct_triangle_index_returned() {
        // Three triangles side by side along X. Aim at the middle one.
        let bvh = bvh_from_tris(&[
            ([-3.,-1.,-2.], [-1.,-1.,-2.], [-2.,1.,-2.]),  // left
            ([-0.5,-1.,-2.],[0.5,-1.,-2.], [0.,1.,-2.]),   // center ← we aim here
            ([ 1.,-1.,-2.], [ 3.,-1.,-2.], [2.,1.,-2.]),   // right
        ]);
        let r   = ray(0., 0., 0.,  0., 0., -1.);
        let hit = intersect_bvh(&bvh, &r, 0.001, 1e30).expect("should hit center triangle");
        let center_tri = &bvh.triangles[hit.triangle_index as usize];
        // The center triangle has vertices near x=0
        assert!(center_tri.v0[0].abs() <= 1.0,
            "hit triangle should be the center one, got v0={:?}", center_tri.v0);
    }

    /// t_max culling: don't return hits farther than the given limit.
    #[test]
    fn test_t_max_culling() {
        let bvh = bvh_from_tris(&[
            ([-1.,-1.,-10.], [1.,-1.,-10.], [0.,1.,-10.]),
        ]);
        let r = ray(0., 0., 0.,  0., 0., -1.);
        // Triangle is at t=10; pass t_max=5
        assert!(intersect_bvh(&bvh, &r, 0.001, 5.0).is_none(),
            "triangle beyond t_max should be culled");
    }

    /// Forces multi-level BVH traversal: 8 triangles so the tree has at least
    /// two levels (MAX_LEAF_TRIS = 4).
    #[test]
    fn test_multi_level_bvh_traversal() {
        let tris: Vec<_> = (0..8).map(|i| {
            let x = i as f32 * 3.0;
            ([x,   -1., -2.], [x+1., -1., -2.], [x+0.5, 1., -2.])
        }).collect();
        let bvh = bvh_from_tris(&tris);
        assert!(bvh.nodes.len() > 1, "should have internal nodes");

        // Aim at each triangle in turn and verify we hit the right one.
        for i in 0..8 {
            let x = i as f32 * 3.0 + 0.5;
            let r   = ray(x, 0., 0.,  0., 0., -1.);
            let hit = intersect_bvh(&bvh, &r, 0.001, 1e30);
            assert!(hit.is_some(), "ray at x={x} should hit triangle {i}");
        }
    }

    /// Two triangles that fit in a single leaf (both ≤ MAX_LEAF_TRIS). The
    /// leaf loop must update `closest` after the first hit so it rejects the
    /// farther triangle, not the nearer one.
    #[test]
    fn test_closest_hit_in_same_leaf() {
        let bvh = bvh_from_tris(&[
            ([-1.,-1.,-1.], [1.,-1.,-1.], [0.,1.,-1.]),  // near  — t = 1
            ([-1.,-1.,-3.], [1.,-1.,-3.], [0.,1.,-3.]),  // far   — t = 3
        ]);
        // Two triangles → one root leaf (MAX_LEAF_TRIS = 4).
        assert_eq!(bvh.nodes.len(), 1, "two triangles should produce a single leaf");

        let r   = ray(0., 0., 0.,  0., 0., -1.);
        let hit = intersect_bvh(&bvh, &r, 0.001, 1e30).expect("should hit");
        assert!(approx_eq(hit.t, 1.0),
            "leaf loop should return the nearer hit (t=1), got t={}", hit.t);
    }

    /// A ray that is not aligned with any world axis exercises all three AABB
    /// slab tests simultaneously during BVH traversal.
    #[test]
    fn test_diagonal_ray_hits_geometry() {
        let bvh = bvh_from_tris(&[
            ([-1.,-1.,-5.], [1.,-1.,-5.], [0.,1.,-5.]),
        ]);
        // Slightly diagonal so the ray still passes through the triangle center.
        let r = ray(0., 0., 0.,  0.1, 0.1, -5.0);
        assert!(intersect_bvh(&bvh, &r, 0.001, 1e30).is_some(),
            "diagonal ray should hit the triangle");
    }

    /// After a hit is found, t_max shrinks. A second triangle that would only
    /// be hit at t > closest must be pruned. Arrange two triangles so one is
    /// clearly closer; then clamp t_max just below the far triangle to confirm
    /// it is never returned.
    #[test]
    fn test_t_max_shrinks_prunes_farther_hits() {
        let bvh = bvh_from_tris(&[
            ([-1.,-1., -2.], [1.,-1., -2.], [0.,1., -2.]),  // near — t=2
            ([-1.,-1.,-10.], [1.,-1.,-10.], [0.,1.,-10.]),  // far  — t=10
        ]);
        let r = ray(0., 0., 0.,  0., 0., -1.);

        // With t_max=5 the far triangle must be culled; near must still hit.
        let hit = intersect_bvh(&bvh, &r, 0.001, 5.0).expect("near triangle should hit");
        assert!(approx_eq(hit.t, 2.0), "should return near hit (t=2), got {}", hit.t);

        // With t_max=1 even the near triangle is culled.
        assert!(intersect_bvh(&bvh, &r, 0.001, 1.0).is_none(),
            "both triangles should be culled when t_max=1");
    }

    /// After a hit, reconstructing the position from barycentric coordinates
    /// must match `origin + t * direction`. This validates that (t, u, v)
    /// are internally consistent.
    #[test]
    fn test_barycentric_reconstruction_matches_hit_position() {
        let v0 = [0., 0., -4.];
        let v1 = [4., 0., -4.];
        let v2 = [0., 4., -4.];
        let bvh = bvh_from_tris(&[(v0, v1, v2)]);
        let r   = ray(1., 1., 0.,  0., 0., -1.);

        let hit = intersect_bvh(&bvh, &r, 0.001, 1e30).expect("should hit triangle");
        let tri = &bvh.triangles[hit.triangle_index as usize];

        // Position from t
        let ray_p = [
            r.origin[0] + hit.t * r.direction[0],
            r.origin[1] + hit.t * r.direction[1],
            r.origin[2] + hit.t * r.direction[2],
        ];
        // Position from barycentrics
        let w = 1. - hit.u - hit.v;
        let bary_p = [
            w * tri.v0[0] + hit.u * tri.v1[0] + hit.v * tri.v2[0],
            w * tri.v0[1] + hit.u * tri.v1[1] + hit.v * tri.v2[1],
            w * tri.v0[2] + hit.u * tri.v1[2] + hit.v * tri.v2[2],
        ];
        for i in 0..3 {
            assert!(approx_eq(ray_p[i], bary_p[i]),
                "axis {i}: ray position {:.5} != barycentric position {:.5}",
                ray_p[i], bary_p[i]);
        }
    }
}
