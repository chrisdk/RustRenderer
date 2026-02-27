//! Ray intersection tests: the two primitives every path tracer needs.
//!
//! This module provides two functions that will be ported verbatim to the
//! WGSL compute shader. The CPU implementations here are the ground truth —
//! if you're debugging the shader and suspect a wrong intersection, reproduce
//! the failing case in Rust first where you have a debugger and unit tests.
//!
//! # Ray–AABB (slab method)
//!
//! An axis-aligned bounding box is the intersection of three "slabs" — the
//! region between a pair of parallel planes. The ray enters each slab at some
//! `t_enter` and exits at some `t_exit`. The ray hits the box if and only if
//! all three entry/exit intervals overlap: `max(t_enters) ≤ min(t_exits)`.
//!
//! # Ray–triangle (Möller–Trumbore)
//!
//! Transforms the intersection problem into barycentric coordinates. The hit
//! point must be inside the triangle (u ≥ 0, v ≥ 0, u + v ≤ 1) and at a
//! positive distance (t > 0). Returns `(t, u, v)` on success; `None` on miss.

use super::math::{cross, dot, sub};

// ============================================================================
// Ray–AABB intersection (slab method)
// ============================================================================

/// Tests whether `ray` intersects an axis-aligned bounding box, returning the
/// entry distance `t` if it does.
///
/// # Arguments
///
/// * `origin`   — ray origin in world space
/// * `inv_dir`  — `1.0 / direction` per component, precomputed by the caller.
///   Reusing this across all AABB tests for the same ray saves three divides
///   per node — a worthwhile saving when traversing thousands of nodes.
/// * `t_min`    — minimum valid distance (typically a small epsilon like 0.001
///   to prevent self-intersection after a bounce)
/// * `t_max`    — maximum valid distance (the current closest hit, updated as
///   we find nearer intersections during traversal)
/// * `aabb_min` — the minimum corner of the box
/// * `aabb_max` — the maximum corner of the box
///
/// Returns `Some(t_entry)` if the ray hits the box within `[t_min, t_max]`,
/// or `None` if it misses (or the box is entirely behind the ray).
///
/// WGSL equivalent:
/// ```wgsl
/// fn ray_aabb(origin: vec3f, inv_dir: vec3f, t_min: f32, t_max: f32,
///             box_min: vec3f, box_max: vec3f) -> f32 {
///     let t1 = (box_min - origin) * inv_dir;
///     let t2 = (box_max - origin) * inv_dir;
///     let t_entry = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
///     let t_exit  = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
///     return select(-1.0, t_entry, t_entry <= t_exit && t_exit >= t_min && t_entry <= t_max);
/// }
/// ```
pub fn ray_aabb(
    origin:   [f32; 3],
    inv_dir:  [f32; 3],
    t_min:    f32,
    t_max:    f32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> Option<f32> {
    let mut t_entry = t_min;
    let mut t_exit  = t_max;

    for i in 0..3 {
        // For each axis, compute where the ray crosses the near and far planes.
        let t1 = (aabb_min[i] - origin[i]) * inv_dir[i];
        let t2 = (aabb_max[i] - origin[i]) * inv_dir[i];
        // t_near is the entry into this slab, t_far is the exit.
        // f32::min/max handle ±infinity correctly for axis-aligned rays.
        t_entry = t_entry.max(t1.min(t2));
        t_exit  = t_exit.min(t1.max(t2));
    }

    if t_entry <= t_exit { Some(t_entry) } else { None }
}

// ============================================================================
// Ray–triangle intersection (Möller–Trumbore)
// ============================================================================

/// Tests whether a ray intersects a triangle, returning `(t, u, v)` on hit.
///
/// Uses the Möller–Trumbore algorithm: it avoids computing the plane equation
/// explicitly and works directly in barycentric coordinates, making it fast
/// and numerically well-behaved.
///
/// # What the return values mean
///
/// * `t` — distance along the ray to the hit point.
///   `hit_point = origin + t * direction`.
/// * `u`, `v` — barycentric coordinates. The hit point expressed in terms of
///   the triangle's three vertices is:
///   `p = (1 - u - v) * v0 + u * v1 + v * v2`
///   Use these to interpolate normals, UVs, and tangents at the hit point.
///
/// # Back-face culling
///
/// If the determinant is ≤ 0, the triangle is either back-facing (the ray
/// comes from behind) or edge-on (degenerate). Both are rejected. For
/// two-sided materials you would test `abs(det)` instead.
///
/// WGSL equivalent: see `intersect_triangle` in `shaders/trace.wgsl`.
pub fn ray_triangle(
    origin:    [f32; 3],
    direction: [f32; 3],
    v0:        [f32; 3],
    v1:        [f32; 3],
    v2:        [f32; 3],
    t_min:     f32,
    t_max:     f32,
) -> Option<(f32, f32, f32)> {
    let e1 = sub(v1, v0);
    let e2 = sub(v2, v0);

    // h = direction × e2; used to compute the determinant.
    let h = cross(direction, e2);
    let det = dot(e1, h);

    // Back-face culling: reject triangles facing away from the ray, and
    // degenerate (near-zero-area) triangles.
    if det < 1e-8 {
        return None;
    }

    let inv_det = 1.0 / det;
    let s = sub(origin, v0);

    // Compute barycentric coordinate u.
    let u = inv_det * dot(s, h);
    if u < 0.0 || u > 1.0 {
        return None;
    }

    // Compute barycentric coordinate v.
    let q = cross(s, e1);
    let v = inv_det * dot(direction, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    // Compute the intersection distance t.
    let t = inv_det * dot(e2, q);
    if t < t_min || t > t_max {
        return None;
    }

    Some((t, u, v))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    fn approx_eq(a: f32, b: f32) -> bool { (a - b).abs() < EPS }

    // ---- ray_aabb tests -----

    /// A ray fired straight into the center of a unit cube should hit it.
    ///
    /// The ray travels along +Z so the caller computes `inv_dir` as
    /// `[1/0, 1/0, 1/1] = [∞, ∞, 1]`. Using `f32::INFINITY` for the zero
    /// components is mandatory — passing `0.0` collapses those slabs to a
    /// degenerate interval and causes a spurious miss.
    #[test]
    fn test_aabb_direct_hit() {
        let t = ray_aabb(
            [0.0, 0.0, -5.0],
            [f32::INFINITY, f32::INFINITY, 1.0],  // inv_dir of direction [0,0,1]
            0.001, 1e30,
            [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0],
        );
        assert!(t.is_some(), "ray should hit the unit cube");
        assert!(approx_eq(t.unwrap(), 4.0), "should enter at t=4, got {:?}", t);
    }

    /// A ray that passes above the box should miss.
    #[test]
    fn test_aabb_miss_above() {
        let t = ray_aabb(
            [0.0, 5.0, -5.0],
            [f32::INFINITY, f32::INFINITY, 1.0],  // direction [0,0,1]
            0.001, 1e30,
            [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0],
        );
        assert!(t.is_none(), "ray above the box should miss");
    }

    /// A ray whose origin is inside the box should still register a hit.
    #[test]
    fn test_aabb_origin_inside_box() {
        // Ray traveling along +X: inv_dir = [1, ∞, ∞].
        let t = ray_aabb(
            [0.0, 0.0, 0.0],
            [1.0, f32::INFINITY, f32::INFINITY],  // direction [1,0,0]
            0.001, 1e30,
            [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0],
        );
        // t_entry is clamped to t_min (the near face is behind us); t_exit = 1.0.
        assert!(t.is_some(), "origin inside box should still be a hit");
    }

    /// A ray moving away from the box (pointing in the wrong direction) should miss.
    #[test]
    fn test_aabb_ray_behind_box() {
        let t = ray_aabb(
            [0.0, 0.0, 5.0],   // behind the box on the +Z side
            [f32::INFINITY, f32::INFINITY, 1.0],  // direction [0,0,1] — moving further away
            0.001, 1e30,
            [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0],
        );
        assert!(t.is_none(), "ray moving away from box should miss");
    }

    /// An axis-aligned ray (traveling exactly along one world axis) should still
    /// work. This exercises the ±infinity path in the slab computation: when a
    /// direction component is zero, `inv_dir` must be `f32::INFINITY`, not `0`.
    #[test]
    fn test_aabb_axis_aligned_ray() {
        // Ray traveling along +Z, passing through the center of the box in XY.
        let t = ray_aabb(
            [0.0, 0.0, -5.0],
            [f32::INFINITY, f32::INFINITY, 1.0],  // direction [0,0,1]
            0.001, 1e30,
            [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0],
        );
        assert!(t.is_some(), "axis-aligned ray through box center should hit");
    }

    /// t_max culling: a box that would be hit at t=14 should be missed when t_max=5.
    #[test]
    fn test_aabb_t_max_culling() {
        let t = ray_aabb(
            [0.0, 0.0, -15.0],
            [f32::INFINITY, f32::INFINITY, 1.0],  // direction [0,0,1]
            0.001, 5.0,  // t_max = 5, box entry at t=14
            [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0],
        );
        assert!(t.is_none(), "box beyond t_max should be culled");
    }

    /// A ray with all three direction components non-zero exercises all three
    /// AABB slabs simultaneously — the case that axis-aligned tests never cover.
    #[test]
    fn test_aabb_diagonal_ray_hit() {
        // Normalized direction [1,1,1]/√3; inv_dir = [√3, √3, √3].
        let s3 = 3.0_f32.sqrt();
        let t = ray_aabb(
            [-5.0, -5.0, -5.0],
            [s3, s3, s3],
            0.001, 1e30,
            [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0],
        );
        // Each slab: t_near = (-1−(−5))×√3 = 4√3; t_far = (1−(−5))×√3 = 6√3
        let expected = 4.0 * s3;
        assert!(t.is_some(), "diagonal ray should hit the unit cube");
        assert!(approx_eq(t.unwrap(), expected),
            "entry should be 4√3 ≈ {expected:.3}, got {:?}", t);
    }

    /// Same diagonal direction but offset so it clears the box.
    #[test]
    fn test_aabb_diagonal_ray_miss() {
        let s3 = 3.0_f32.sqrt();
        // Displaced far in Z — the Z slab is entirely behind when all slabs combined.
        let t = ray_aabb(
            [-5.0, -5.0, 10.0],   // Z offset ensures Z slab exits behind origin
            [s3, s3, s3],
            0.001, 1e30,
            [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0],
        );
        assert!(t.is_none(), "offset diagonal ray should miss the box");
    }

    /// A ray traveling in −Z (the default camera direction) uses negative
    /// inv_dir components and must still hit correctly.
    #[test]
    fn test_aabb_negative_direction_ray() {
        // Direction [0,0,−1]: inv_dir = [∞, ∞, −1].
        let t = ray_aabb(
            [0.0, 0.0, 5.0],
            [f32::INFINITY, f32::INFINITY, -1.0],
            0.001, 1e30,
            [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0],
        );
        assert!(t.is_some(), "ray in −Z should hit the box");
        // Entry: Z slab t1=(−1−5)×(−1)=6, t2=(1−5)×(−1)=4; min=4 → entry at t=4.
        assert!(approx_eq(t.unwrap(), 4.0), "should enter at t=4, got {:?}", t);
    }

    // ---- ray_triangle tests ----

    /// A ray fired perpendicularly at a flat triangle should hit at t = distance.
    #[test]
    fn test_triangle_perpendicular_hit() {
        // Triangle in the z=-2 plane
        let v0 = [-1.0, -1.0, -2.0];
        let v1 = [ 1.0, -1.0, -2.0];
        let v2 = [ 0.0,  1.0, -2.0];
        let hit = ray_triangle(
            [0.0, 0.0, 0.0], [0.0, 0.0, -1.0],
            v0, v1, v2, 0.001, 1e30,
        );
        assert!(hit.is_some(), "ray should hit the triangle");
        let (t, _u, _v) = hit.unwrap();
        assert!(approx_eq(t, 2.0), "t should be 2.0, got {t}");
    }

    /// A ray aimed to the side of the triangle should miss.
    #[test]
    fn test_triangle_side_miss() {
        let v0 = [-1.0, -1.0, -2.0];
        let v1 = [ 1.0, -1.0, -2.0];
        let v2 = [ 0.0,  1.0, -2.0];
        let hit = ray_triangle(
            [5.0, 0.0, 0.0], [0.0, 0.0, -1.0],
            v0, v1, v2, 0.001, 1e30,
        );
        assert!(hit.is_none(), "ray to the side should miss");
    }

    /// A ray hitting the exact midpoint of one edge should still register.
    #[test]
    fn test_triangle_edge_hit() {
        // Hit the midpoint of the bottom edge (v0–v1): u=0.5, v=0, w=0.5
        let v0 = [0.0, 0.0, -2.0];
        let v1 = [2.0, 0.0, -2.0];
        let v2 = [1.0, 2.0, -2.0];
        let hit = ray_triangle(
            [1.0, 0.0, 0.0], [0.0, 0.0, -1.0],
            v0, v1, v2, 0.001, 1e30,
        );
        assert!(hit.is_some(), "edge midpoint hit should succeed");
    }

    /// A ray coming from behind the triangle (back face) should be culled.
    #[test]
    fn test_triangle_back_face_culled() {
        let v0 = [-1.0, -1.0, -2.0];
        let v1 = [ 1.0, -1.0, -2.0];
        let v2 = [ 0.0,  1.0, -2.0];
        // Shoot from the back side (positive Z, traveling in +Z direction)
        let hit = ray_triangle(
            [0.0, 0.0, -4.0], [0.0, 0.0, 1.0],
            v0, v1, v2, 0.001, 1e30,
        );
        assert!(hit.is_none(), "back-face ray should be culled");
    }

    /// A ray parallel to the triangle plane should miss.
    ///
    /// The triangle lies in the XZ plane (Y = 0). A ray traveling along +X at
    /// Y = 1 is parallel to that plane — it will never cross Y = 0, so the
    /// determinant in Möller–Trumbore is zero and the function returns None.
    #[test]
    fn test_triangle_parallel_ray() {
        let v0 = [-1.0, 0.0, -1.0];
        let v1 = [ 1.0, 0.0, -1.0];
        let v2 = [ 0.0, 0.0,  1.0];
        // Ray at Y=1, traveling along +X — parallel to the XZ plane.
        let hit = ray_triangle(
            [-5.0, 1.0, 0.0], [1.0, 0.0, 0.0],
            v0, v1, v2, 0.001, 1e30,
        );
        assert!(hit.is_none(), "parallel ray should miss");
    }

    /// A hit at t = 0.5 should be rejected when t_min = 1.0.
    #[test]
    fn test_triangle_t_below_t_min_rejected() {
        let v0 = [-1.0, -1.0, -0.5];
        let v1 = [ 1.0, -1.0, -0.5];
        let v2 = [ 0.0,  1.0, -0.5];
        let hit = ray_triangle(
            [0.0, 0.0, 0.0], [0.0, 0.0, -1.0],
            v0, v1, v2,
            1.0,  // t_min = 1.0, hit would be at t = 0.5
            1e30,
        );
        assert!(hit.is_none(), "hit before t_min should be rejected");
    }

    /// A ray aimed directly at a vertex (u=0, v=0 → w=1) must register a hit.
    /// Vertices are boundary cases of the u/v guards in Möller–Trumbore.
    #[test]
    fn test_triangle_vertex_hit() {
        let v0 = [0.0, 0.0, -2.0];
        let v1 = [2.0, 0.0, -2.0];
        let v2 = [0.0, 2.0, -2.0];
        // Ray aimed straight at v0 — barycentrics should be u=0, v=0.
        let hit = ray_triangle(
            [0.0, 0.0, 0.0], [0.0, 0.0, -1.0],
            v0, v1, v2, 0.001, 1e30,
        );
        assert!(hit.is_some(), "ray at v0 should hit the triangle");
        let (t, u, v) = hit.unwrap();
        assert!(approx_eq(t, 2.0), "t should be 2.0, got {t}");
        assert!(approx_eq(u, 0.0) && approx_eq(v, 0.0),
            "vertex hit should have u=v=0, got u={u} v={v}");
    }

    /// A hit at t=10 should be rejected when t_max=5 (symmetry with the
    /// t_min test above).
    #[test]
    fn test_triangle_t_above_t_max_rejected() {
        let v0 = [-1.0, -1.0, -10.0];
        let v1 = [ 1.0, -1.0, -10.0];
        let v2 = [ 0.0,  1.0, -10.0];
        let hit = ray_triangle(
            [0.0, 0.0, 0.0], [0.0, 0.0, -1.0],
            v0, v1, v2,
            0.001,
            5.0,   // t_max = 5, triangle is at t = 10
        );
        assert!(hit.is_none(), "hit beyond t_max should be rejected");
    }

    /// Collinear vertices produce a zero-area triangle. The determinant in
    /// Möller–Trumbore is zero, triggering the `det < 1e-8` guard.
    #[test]
    fn test_triangle_degenerate_collinear() {
        // All three vertices on the same line — zero area, no valid hit.
        let v0 = [0.0, 0.0, -2.0];
        let v1 = [1.0, 0.0, -2.0];
        let v2 = [2.0, 0.0, -2.0]; // collinear with v0–v1
        let hit = ray_triangle(
            [1.0, 0.0, 0.0], [0.0, 0.0, -1.0],
            v0, v1, v2, 0.001, 1e30,
        );
        assert!(hit.is_none(), "degenerate (collinear) triangle should return None");
    }

    /// Barycentric coordinates must satisfy u ≥ 0, v ≥ 0, u + v ≤ 1.
    #[test]
    fn test_triangle_barycentric_coords_valid() {
        let v0 = [0.0, 0.0, -3.0];
        let v1 = [4.0, 0.0, -3.0];
        let v2 = [0.0, 4.0, -3.0];
        // Aim at a few interior points and verify barycentric coords are valid.
        for &(ox, oy) in &[(0.5_f32, 0.5_f32), (1.0, 1.0), (0.1, 2.0)] {
            if let Some((_t, u, v)) = ray_triangle(
                [ox, oy, 0.0], [0.0, 0.0, -1.0],
                v0, v1, v2, 0.001, 1e30,
            ) {
                assert!(u >= 0.0 && v >= 0.0 && u + v <= 1.0 + EPS,
                    "invalid barycentrics at ({ox},{oy}): u={u}, v={v}");
            }
        }
    }
}
