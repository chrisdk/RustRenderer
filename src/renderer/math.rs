//! Vector math utilities used by the CPU traversal and — critically — ported
//! verbatim to WGSL for the GPU path tracing shader.
//!
//! Keeping the CPU and GPU implementations in sync is the whole point. If you
//! change a formula here, change it in the shader too, and vice versa. The
//! CPU unit tests serve as the ground truth for verifying the shader.
//!
//! All functions operate on plain `[f32; 3]` arrays to stay consistent with
//! the rest of the codebase. Each function has a direct WGSL equivalent noted
//! in its doc comment.

/// Adds two vectors component-wise. WGSL: `a + b`.
#[inline]
pub fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Subtracts `b` from `a` component-wise. WGSL: `a - b`.
#[inline]
pub fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Multiplies each component of `v` by scalar `s`. WGSL: `v * s`.
#[inline]
pub fn scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Fused multiply-add: `a + b * s`. Handy for "advance ray by t" calculations.
/// WGSL: `a + b * s`.
#[inline]
pub fn mad(a: [f32; 3], b: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]
}

/// Dot product: a scalar measuring how much two vectors point in the same
/// direction. If both are unit vectors, the result is the cosine of the angle
/// between them. WGSL: `dot(a, b)`.
#[inline]
pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product: a vector perpendicular to both `a` and `b`. Its length
/// equals the area of the parallelogram they span. WGSL: `cross(a, b)`.
#[inline]
pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Returns the squared length of a vector. Cheaper than `length` when you
/// only need to compare distances (avoids a square root). WGSL: `dot(v, v)`.
#[inline]
pub fn length_sq(v: [f32; 3]) -> f32 {
    dot(v, v)
}

/// Returns the length (Euclidean norm) of a vector. WGSL: `length(v)`.
#[inline]
pub fn length(v: [f32; 3]) -> f32 {
    length_sq(v).sqrt()
}

/// Scales a vector to unit length. The result has length 1 but points in the
/// same direction. Undefined (returns NaN) if `v` is the zero vector — avoid
/// normalizing zero-length directions. WGSL: `normalize(v)`.
#[inline]
pub fn normalize(v: [f32; 3]) -> [f32; 3] {
    scale(v, 1.0 / length(v))
}

/// Reflects incident vector `i` off a surface with unit normal `n`.
///
/// Imagine light hitting a mirror: the angle of incidence equals the angle of
/// reflection. `i` is the incoming direction, `n` is the surface normal, and
/// the result is the outgoing (reflected) direction.
///
/// Formula: `i - 2 * dot(i, n) * n`. WGSL: `reflect(i, n)` (same formula).
///
/// Both `i` and `n` should be normalized for a physically correct result.
#[inline]
pub fn reflect(i: [f32; 3], n: [f32; 3]) -> [f32; 3] {
    sub(i, scale(n, 2.0 * dot(i, n)))
}

/// Schlick's approximation for Fresnel reflectance.
///
/// Physically, highly conductive materials (metals) reflect more light at
/// glancing angles than at head-on angles. `f0` is the base reflectance at
/// normal incidence (looking straight at the surface), and `cos_theta` is the
/// cosine of the angle between the view direction and the surface normal.
///
/// At `cos_theta = 1.0` (looking straight on), the result equals `f0`.
/// At `cos_theta = 0.0` (grazing angle), the result approaches white [1,1,1].
///
/// WGSL equivalent:
/// ```wgsl
/// fn fresnel_schlick(cos_theta: f32, f0: vec3f) -> vec3f {
///     return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
/// }
/// ```
#[inline]
pub fn fresnel_schlick(cos_theta: f32, f0: [f32; 3]) -> [f32; 3] {
    let t = (1.0 - cos_theta).powi(5);
    [
        f0[0] + (1.0 - f0[0]) * t,
        f0[1] + (1.0 - f0[1]) * t,
        f0[2] + (1.0 - f0[2]) * t,
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool { (a - b).abs() < EPS }
    fn vec_eq(a: [f32; 3], b: [f32; 3]) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    #[test]
    fn test_cross_product_standard_basis() {
        // The right-hand rule: X × Y = Z, Y × Z = X, Z × X = Y.
        assert!(vec_eq(cross([1.,0.,0.], [0.,1.,0.]), [0.,0.,1.]));
        assert!(vec_eq(cross([0.,1.,0.], [0.,0.,1.]), [1.,0.,0.]));
        assert!(vec_eq(cross([0.,0.,1.], [1.,0.,0.]), [0.,1.,0.]));
    }

    #[test]
    fn test_cross_product_is_perpendicular() {
        // The cross product of any two vectors should be perpendicular to both.
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let c = cross(a, b);
        assert!(approx_eq(dot(c, a), 0.0), "cross(a,b) · a should be 0");
        assert!(approx_eq(dot(c, b), 0.0), "cross(a,b) · b should be 0");
    }

    #[test]
    fn test_normalize_unit_vector_unchanged() {
        let v = normalize([1.0, 0.0, 0.0]);
        assert!(approx_eq(length(v), 1.0));
        assert!(vec_eq(v, [1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_normalize_arbitrary_vector() {
        // Any non-zero vector normalized should have length 1.
        let v = normalize([3.0, 4.0, 0.0]);
        assert!(approx_eq(length(v), 1.0));
        assert!(approx_eq(v[0], 0.6));
        assert!(approx_eq(v[1], 0.8));
    }

    #[test]
    fn test_reflect_horizontal_ray_off_y_normal() {
        // A ray going diagonally down-right reflecting off a flat floor (Y+ normal)
        // should come out diagonally up-right.
        let incident  = normalize([1.0, -1.0, 0.0]);
        let normal    = [0.0, 1.0, 0.0];
        let reflected = reflect(incident, normal);
        let expected  = normalize([1.0, 1.0, 0.0]);
        assert!(vec_eq(reflected, expected),
            "reflected = {:?}, expected {:?}", reflected, expected);
    }

    #[test]
    fn test_fresnel_at_normal_incidence_equals_f0() {
        // Looking straight at a surface (cos_theta = 1) → result should be f0.
        let f0 = [0.04, 0.04, 0.04];
        let result = fresnel_schlick(1.0, f0);
        assert!(vec_eq(result, f0),
            "fresnel at cos=1 should equal f0, got {:?}", result);
    }

    #[test]
    fn test_fresnel_at_grazing_angle_approaches_white() {
        // At 90° (cos_theta = 0) all surfaces look like a mirror.
        let f0     = [0.04, 0.04, 0.04];
        let result = fresnel_schlick(0.0, f0);
        assert!(vec_eq(result, [1.0, 1.0, 1.0]),
            "fresnel at cos=0 should be [1,1,1], got {:?}", result);
    }

    #[test]
    fn test_mad_correctness() {
        // mad(a, b, s) = a + b*s
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 0.0, -1.0];
        let result = mad(a, b, 2.0);
        assert!(vec_eq(result, [3.0, 2.0, 1.0]));
    }
}
