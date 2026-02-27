//! The camera: where rays come from.
//!
//! In path tracing, rendering a frame means shooting one (or many) rays
//! through each pixel, bouncing them around the scene, and averaging up
//! what they hit. This module handles step one: given a pixel coordinate,
//! produce the initial ray. Everything else is someone else's problem.
//!
//! # Coordinate system
//!
//! We follow the GLTF / WebGPU convention: **Y-up, right-handed**. At the
//! default orientation (yaw = 0, pitch = 0), the camera looks along **−Z**,
//! with +X to the right and +Y upward.
//!
//! # FPS-style navigation
//!
//! The camera is parameterized as a position + yaw + pitch, exactly like a
//! first-person shooter. Yaw rotates left/right; pitch tilts up/down. No roll
//! — the camera always stays level. This gives intuitive mouse-look controls
//! without the complexity of a full quaternion orientation.

use std::f32::consts::FRAC_PI_2;

// ============================================================================
// Constants
// ============================================================================

/// The maximum pitch angle, just shy of looking straight up (or down).
///
/// Clamping pitch to ±MAX_PITCH prevents the camera from flipping upside-down
/// when the user drags the mouse too far. π/2 − 0.001 lets you get *very*
/// close to vertical without the math degenerating.
const MAX_PITCH: f32 = FRAC_PI_2 - 0.001;

// ============================================================================
// Ray
// ============================================================================

/// A ray: an origin point plus a normalized direction vector.
///
/// Rays are the atomic unit of path tracing. The camera fires primary rays
/// through each pixel; after each surface hit, the integrator spawns secondary
/// rays to sample indirect lighting. The direction is always normalized (length
/// = 1) so that "the point at distance t along the ray" is simply
/// `origin + t * direction`.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    /// The starting point of the ray in world space.
    pub origin: [f32; 3],

    /// The direction the ray travels. Always a unit vector.
    pub direction: [f32; 3],
}

// ============================================================================
// Camera
// ============================================================================

/// A first-person camera that generates rays for path tracing.
///
/// The camera is kept deliberately simple: a world-space position and two
/// angles (yaw, pitch) describe where it is and where it's pointing.
/// Call [`Camera::ray`] for any pixel to get the ray to trace.
///
/// # Example
///
/// ```rust
/// let mut cam = Camera::new([0.0, 1.0, 5.0], PI / 3.0, 16.0 / 9.0);
/// // look slightly to the right and move forward a bit
/// cam.pan(0.3, 0.0);
/// cam.translate(1.0, 0.0, 0.0);
/// // trace the center pixel
/// let ray = cam.ray(0.5, 0.5);
/// ```
pub struct Camera {
    /// Camera position in world space.
    pub position: [f32; 3],

    /// Horizontal rotation in radians. 0 = looking along −Z. Positive values
    /// rotate clockwise when viewed from above (i.e. turning right).
    pub yaw: f32,

    /// Vertical tilt in radians. 0 = horizontal. Positive = looking up.
    /// Always clamped to [−MAX_PITCH, MAX_PITCH] — this is not the camera
    /// that films action movies with wild dutch angles.
    pub pitch: f32,

    /// Vertical field of view in radians. π/3 (60°) feels natural; π/2 (90°)
    /// gives a wider game-style view. Going beyond that starts to look fishy.
    pub vfov: f32,

    /// Render target width divided by height. Prevents the scene from looking
    /// squished when the canvas isn't square.
    pub aspect: f32,
}

impl Camera {
    /// Creates a new camera at `position`, facing −Z (yaw = 0, pitch = 0).
    ///
    /// `vfov` is the vertical field of view in radians; `aspect` is
    /// width / height of the render target.
    pub fn new(position: [f32; 3], vfov: f32, aspect: f32) -> Self {
        Camera { position, yaw: 0.0, pitch: 0.0, vfov, aspect }
    }

    /// Generates a world-space ray for the pixel at normalized screen
    /// coordinates `(u, v)`.
    ///
    /// `u` and `v` are both in \[0, 1\]. The origin is the **top-left** corner
    /// of the image: `u` increases to the right, `v` increases downward,
    /// matching standard raster convention. The center of the image is
    /// `(0.5, 0.5)`.
    ///
    /// For anti-aliasing, jitter `u` and `v` by a sub-pixel random offset
    /// before calling this — the Monte Carlo math takes care of the rest.
    pub fn ray(&self, u: f32, v: f32) -> Ray {
        // The "virtual film plane" sits one unit in front of the camera. Its
        // half-height is tan(vfov/2) and its half-width is aspect × half-height.
        let half_h = (self.vfov * 0.5).tan();
        let half_w = self.aspect * half_h;

        let (fwd, rgt, up) = self.basis();

        // Map pixel (u, v) to a point on the film plane.
        // u ∈ [0,1] → film_x ∈ [−half_w, +half_w]
        // v ∈ [0,1] → film_y ∈ [+half_h, −half_h]  (flipped: v=0 is the top = +Y)
        let film_x = (2.0 * u - 1.0) * half_w;
        let film_y = (1.0 - 2.0 * v) * half_h;

        let dir = [
            fwd[0] + film_x * rgt[0] + film_y * up[0],
            fwd[1] + film_x * rgt[1] + film_y * up[1],
            fwd[2] + film_x * rgt[2] + film_y * up[2],
        ];

        Ray { origin: self.position, direction: normalize(dir) }
    }

    /// Rotates the camera: `dyaw` radians horizontally, `dpitch` vertically.
    ///
    /// Pitch is clamped after the update so the camera can never look fully
    /// straight up or down (and definitely can't look behind itself over the
    /// top of its own head).
    pub fn pan(&mut self, dyaw: f32, dpitch: f32) {
        self.yaw   += dyaw;
        self.pitch  = (self.pitch + dpitch).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// Moves the camera in its local coordinate frame.
    ///
    /// - `fwd` — distance along the look direction (negative = step back)
    /// - `right` — distance along the right vector (negative = strafe left)
    /// - `up` — distance along the camera's up vector (negative = sink down)
    ///
    /// All three axes are applied simultaneously, so diagonal movement is
    /// possible without calling this twice.
    pub fn translate(&mut self, fwd: f32, right: f32, up: f32) {
        let (f, r, u_vec) = self.basis();
        for i in 0..3 {
            self.position[i] += fwd * f[i] + right * r[i] + up * u_vec[i];
        }
    }

    /// Returns the camera's three orthonormal basis vectors: (forward, right, up).
    ///
    /// All three are derived analytically from yaw and pitch, so they're
    /// always in sync with no risk of accumulated floating-point drift.
    ///
    /// ```text
    /// forward = [sin(yaw)·cos(pitch),  sin(pitch),  −cos(yaw)·cos(pitch)]
    /// right   = [cos(yaw),             0,            sin(yaw)            ]
    /// up      = cross(right, forward)
    /// ```
    ///
    /// The right vector has no Y component by design — strafe is always
    /// horizontal, unaffected by pitch. This is what makes it feel like an
    /// FPS rather than a space sim.
    fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();

        let fwd   = [sy * cp, sp, -cy * cp];
        let right = [cy, 0.0, sy];
        // forward ⊥ right (you can verify: their dot product is zero).
        // Since both are unit vectors and mutually perpendicular, their cross
        // product is automatically a unit vector — no normalize needed.
        let up = cross(right, fwd);

        (fwd, right, up)
    }
}

// ============================================================================
// Private math helpers
// ============================================================================

/// Scales a vector to unit length.
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Returns the cross product of two vectors: a vector perpendicular to both.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]
}

/// Returns the dot product: a scalar measuring how much two vectors align.
/// Zero means perpendicular; 1 means parallel (for unit vectors).
#[allow(dead_code)] // used in tests and will be needed by the integrator
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn vec_approx_eq(a: [f32; 3], b: [f32; 3]) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    /// The center pixel of a default camera must produce a ray pointing exactly
    /// down −Z. If this fails, everything downstream is wrong.
    #[test]
    fn test_center_ray_faces_negative_z() {
        let cam = Camera::new([0.0, 0.0, 0.0], FRAC_PI_2, 1.0);
        let ray = cam.ray(0.5, 0.5);
        assert!(
            vec_approx_eq(ray.direction, [0.0, 0.0, -1.0]),
            "center ray should be (0, 0, −1), got {:?}", ray.direction,
        );
    }

    /// Every ray direction must be a unit vector, regardless of where in the
    /// frame it lands. A non-normalized direction would give incorrect
    /// intersection distances and break all the lighting math.
    #[test]
    fn test_ray_direction_is_normalized() {
        let cam = Camera::new([3.0, 1.0, -2.0], std::f32::consts::PI / 3.0, 16.0 / 9.0);
        let samples = [
            (0.0_f32, 0.0_f32), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
            (0.5, 0.5), (0.25, 0.75), (0.1, 0.9),
        ];
        for (u, v) in samples {
            let r   = cam.ray(u, v);
            let len = dot(r.direction, r.direction).sqrt();
            assert!(
                approx_eq(len, 1.0),
                "ray({u}, {v}) direction has length {len}, expected 1.0",
            );
        }
    }

    /// With a 90° vertical FOV and a square (1:1) aspect ratio, the right-edge
    /// center ray should be exactly 45° from the forward direction — i.e.
    /// (√2/2, 0, −√2/2). This confirms the FOV and film-plane math are correct.
    #[test]
    fn test_90_degree_fov_edge_angles() {
        let cam = Camera::new([0.0, 0.0, 0.0], FRAC_PI_2, 1.0);

        // Right-edge center: raw dir = forward + 1·right = (0,0,−1) + (1,0,0) = (1,0,−1)
        let s2 = 2.0_f32.sqrt() / 2.0;
        let ray = cam.ray(1.0, 0.5);
        assert!(
            vec_approx_eq(ray.direction, [s2, 0.0, -s2]),
            "right-edge ray: {:?}, expected ({s2}, 0, {s2})", ray.direction,
        );

        // Top-right corner: raw dir = (1, 1, −1) → length √3 → each component 1/√3
        let s3 = 1.0 / 3.0_f32.sqrt();
        let corner = cam.ray(1.0, 0.0);
        assert!(
            vec_approx_eq(corner.direction, [s3, s3, -s3]),
            "top-right corner ray: {:?}, expected ({s3}, {s3}, {s3})", corner.direction,
        );
    }

    /// After a 90° yaw to the right, the center ray should point along +X.
    #[test]
    fn test_yaw_rotates_view_direction() {
        let mut cam = Camera::new([0.0, 0.0, 0.0], FRAC_PI_2, 1.0);
        cam.pan(FRAC_PI_2, 0.0);
        let ray = cam.ray(0.5, 0.5);
        assert!(
            vec_approx_eq(ray.direction, [1.0, 0.0, 0.0]),
            "after 90° yaw, center ray should be (1, 0, 0), got {:?}", ray.direction,
        );
    }

    /// Translating forward at the default orientation (facing −Z) should move
    /// the camera position along −Z.
    #[test]
    fn test_translate_forward_moves_along_negative_z() {
        let mut cam = Camera::new([5.0, 1.0, 3.0], FRAC_PI_2, 1.0);
        cam.translate(2.0, 0.0, 0.0);
        assert!(
            vec_approx_eq(cam.position, [5.0, 1.0, 1.0]),
            "after translating 2 units forward, position should be (5, 1, 1), got {:?}", cam.position,
        );
    }

    /// No matter how violently the user swipes the mouse, pitch must never
    /// exceed MAX_PITCH. Flipping the camera upside-down would be disorienting
    /// and also slightly break the basis math.
    #[test]
    fn test_pitch_is_clamped() {
        let mut cam = Camera::new([0.0, 0.0, 0.0], FRAC_PI_2, 1.0);

        cam.pan(0.0, 999.0);  // look way, way up
        assert!(
            approx_eq(cam.pitch, MAX_PITCH),
            "upward pitch should be clamped to MAX_PITCH ({MAX_PITCH}), got {}", cam.pitch,
        );

        cam.pan(0.0, -999.0); // look way, way down
        assert!(
            approx_eq(cam.pitch, -MAX_PITCH),
            "downward pitch should be clamped to −MAX_PITCH ({MAX_PITCH}), got {}", cam.pitch,
        );
    }
}
