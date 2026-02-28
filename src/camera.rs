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
// CameraUniform — GPU-side representation
// ============================================================================

/// A GPU-ready snapshot of the camera, sized and padded for a WGSL uniform buffer.
///
/// WGSL requires `vec3<f32>` fields to be 16-byte aligned (same as `vec4`).
/// We satisfy that by adding a `_pad` float after each vec3, giving four tidy
/// 16-byte slots and a total size of 64 bytes.
///
/// The shader can generate the primary ray for pixel `(u, v)` as:
/// ```text
/// dir = fwd + (2u − 1) · rgt_scaled + (1 − 2v) · up_scaled
/// ray = { origin: position, direction: normalize(dir) }
/// ```
/// The `rgt_scaled` / `up_scaled` vectors already have the film-plane half-width
/// and half-height baked in, so the shader skips that arithmetic.
///
/// WGSL binding: `@group(1) @binding(0) var<uniform> camera: Camera`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// World-space camera position. `_pad0` pads to 16 bytes.
    pub position:   [f32; 3],
    pub _pad0:      f32,

    /// Unit forward vector (the direction the camera is pointing).
    pub fwd:        [f32; 3],
    pub _pad1:      f32,

    /// Right vector pre-scaled by `half_width = aspect × tan(vfov / 2)`.
    pub rgt_scaled: [f32; 3],
    pub _pad2:      f32,

    /// Up vector pre-scaled by `half_height = tan(vfov / 2)`.
    pub up_scaled:  [f32; 3],
    pub _pad3:      f32,
}

// ============================================================================
// RasterCameraUniform — matrices for the rasterizer pipeline
// ============================================================================

/// A GPU-ready camera uniform for the rasterizing preview renderer.
///
/// The path-tracer uses ray generation (origin + film-plane vectors); the
/// rasterizer uses the classical view-projection matrix pair. Both describe
/// the same camera — just different math on the same yaw/pitch/position state.
///
/// Layout: 144 bytes — two 4×4 matrices (128 B) + world position (12 B) + 4 B pad.
///
/// The WGSL vertex shader computes clip position as:
/// ```wgsl
/// let clip_pos = proj * view * model * vec4<f32>(position, 1.0);
/// ```
/// The fragment shader needs the world-space camera position to compute the
/// view direction for specular highlights — it's cheaper to pass it explicitly
/// than to recover it from the view matrix inverse.
///
/// WGSL binding: `@group(0) @binding(0) var<uniform> cam: RasterCamera`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RasterCameraUniform {
    /// View matrix: transforms world space → camera space.
    /// Column-major, so `view[col][row]` matches WGSL's `mat4x4` layout.
    pub view: [[f32; 4]; 4],

    /// Projection matrix: transforms camera space → clip space.
    /// Standard perspective projection (right-handed, WebGPU NDC z ∈ [0, 1]).
    pub proj: [[f32; 4]; 4],

    /// Camera position in world space, for view-direction computation in the
    /// fragment shader. Followed by a padding float to reach 144 bytes
    /// (the `vec3<f32>` in WGSL has 16-byte alignment, so the 4-byte pad
    /// keeps the struct a multiple of 16 bytes).
    pub cam_pos: [f32; 3],
    pub _pad: f32,
}

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
/// use render::camera::Camera;
/// use std::f32::consts::PI;
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

    /// Produces a [`CameraUniform`] ready for upload to a GPU uniform buffer.
    ///
    /// Call this whenever the camera moves and pass the result to
    /// `Renderer::upload_camera`. The heavy-ish trig (sin/cos for yaw and pitch,
    /// one tangent for the FOV) happens here on the CPU so the shader stays lean.
    pub fn to_uniform(&self) -> CameraUniform {
        let half_h = (self.vfov * 0.5).tan();
        let half_w = self.aspect * half_h;
        let (fwd, rgt, up) = self.basis();

        CameraUniform {
            position:   self.position,
            _pad0:      0.0,
            fwd,
            _pad1:      0.0,
            rgt_scaled: [rgt[0] * half_w, rgt[1] * half_w, rgt[2] * half_w],
            _pad2:      0.0,
            up_scaled:  [up[0] * half_h, up[1] * half_h, up[2] * half_h],
            _pad3:      0.0,
        }
    }

    /// Produces a [`RasterCameraUniform`] for the rasterizing preview pipeline.
    ///
    /// Builds the classical view + projection matrix pair from the same
    /// yaw/pitch/position state used by `to_uniform()`. The path tracer and
    /// rasterizer always stay in sync because they both derive from the same
    /// Camera fields.
    ///
    /// The view matrix is the inverse of the camera-to-world transform.
    /// Because the basis is orthonormal, the inverse is cheap: transpose the
    /// 3×3 rotation block and negate the translation by dotting.
    ///
    /// The projection uses WebGPU NDC convention: x,y ∈ [−1, 1], z ∈ [0, 1].
    pub fn to_raster_uniform(&self) -> RasterCameraUniform {
        let (fwd, rgt, up) = self.basis();
        let pos = self.position;

        // ---- View matrix (world → camera space) ----
        //
        // For an orthonormal frame {rgt, up, −fwd}, the inverse is the
        // transpose of the rotation block plus a negated-dot-product
        // translation. Stored column-major: m[col][row].
        //
        //   Row 0 (right):    [ rgt.x  rgt.y  rgt.z  −(rgt · pos) ]
        //   Row 1 (up):       [  up.x   up.y   up.z  −( up · pos) ]
        //   Row 2 (−forward): [−fwd.x −fwd.y −fwd.z  +(fwd · pos) ]
        //   Row 3:            [     0      0      0              1 ]
        let view = [
            [rgt[0],  up[0],  -fwd[0], 0.0],  // column 0
            [rgt[1],  up[1],  -fwd[1], 0.0],  // column 1
            [rgt[2],  up[2],  -fwd[2], 0.0],  // column 2
            [                                  // column 3 (translation)
                -(rgt[0]*pos[0] + rgt[1]*pos[1] + rgt[2]*pos[2]),
                -( up[0]*pos[0] +  up[1]*pos[1] +  up[2]*pos[2]),
                  fwd[0]*pos[0] + fwd[1]*pos[1] + fwd[2]*pos[2],
                1.0,
            ],
        ];

        // ---- Projection matrix (camera → clip space) ----
        //
        // Standard right-handed perspective, WebGPU NDC (z ∈ [0, 1]).
        // Camera looks along −Z, so the near plane is at z = −near.
        //
        //   | 1/(a·h)    0         0           0        |
        //   |    0      1/h        0           0        |
        //   |    0       0      −f/(f−n)  −f·n/(f−n)   |
        //   |    0       0        −1           0        |
        //
        // where h = tan(vfov/2),  a = aspect,  n = near,  f = far.
        let near: f32 = 0.01;
        let far:  f32 = 1000.0;
        let inv_h = 1.0 / (self.vfov * 0.5).tan();  // 1 / tan(vfov/2)
        let inv_w = inv_h / self.aspect;             // 1 / (aspect · tan(vfov/2))
        let range = far / (far - near);              // f / (f − n)

        let proj = [
            [inv_w, 0.0,    0.0,        0.0],  // column 0
            [0.0,   inv_h,  0.0,        0.0],  // column 1
            [0.0,   0.0,   -range,     -1.0],  // column 2
            [0.0,   0.0,   -range * near, 0.0], // column 3
        ];

        RasterCameraUniform { view, proj, cam_pos: self.position, _pad: 0.0 }
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

    /// With a 2:1 aspect ratio the horizontal FOV must be wider than the
    /// vertical FOV. The right-edge ray's X deflection must exceed the
    /// top-edge ray's Y deflection.
    #[test]
    fn test_aspect_ratio_widens_horizontal_fov() {
        let cam = Camera::new([0., 0., 0.], FRAC_PI_2, 2.0); // 2:1 aspect

        // Right-edge center ray: film_x = half_w = 2×tan(45°) = 2; film_y = 0.
        // Raw dir = (0,0,−1) + 2×(1,0,0) = (2,0,−1); normalized x = 2/√5 ≈ 0.894
        let right_edge = cam.ray(1.0, 0.5);

        // Top-edge center ray: film_x = 0; film_y = half_h = 1.
        // Raw dir = (0,0,−1) + (0,1,0); normalized y = 1/√2 ≈ 0.707
        let top_edge = cam.ray(0.5, 0.0);

        assert!(
            right_edge.direction[0] > top_edge.direction[1],
            "2:1 aspect: horizontal deflection ({:.3}) should exceed vertical ({:.3})",
            right_edge.direction[0], top_edge.direction[1],
        );
    }

    /// translate() accepts independent forward, right, and up scalars.
    /// Verify that the right and up axes move the camera in the expected
    /// world-space directions at the default orientation.
    #[test]
    fn test_translate_right_and_up() {
        // Default orientation: facing −Z, right = +X, up = +Y.
        let mut cam = Camera::new([0., 0., 0.], FRAC_PI_2, 1.0);
        cam.translate(0.0, 3.0, 0.0); // strafe right
        assert!(
            vec_approx_eq(cam.position, [3.0, 0.0, 0.0]),
            "strafing right should move +X, got {:?}", cam.position,
        );

        let mut cam = Camera::new([0., 0., 0.], FRAC_PI_2, 1.0);
        cam.translate(0.0, 0.0, 2.0); // move up
        assert!(
            vec_approx_eq(cam.position, [0.0, 2.0, 0.0]),
            "moving up should move +Y, got {:?}", cam.position,
        );
    }

    /// At every orientation, forward/right/up must be mutually perpendicular
    /// unit vectors. Non-orthonormal bases skew the image and break intersection
    /// distances.
    #[test]
    fn test_basis_is_orthonormal() {
        let cases = [
            (0.0_f32, 0.0_f32),  // default
            (FRAC_PI_2, 0.0),    // 90° yaw
            (0.0, 1.0),          // steep upward pitch
            (0.0, -1.0),         // steep downward pitch
            (0.7, 0.4),          // arbitrary
        ];
        for (yaw, pitch) in cases {
            let mut cam = Camera::new([0., 0., 0.], FRAC_PI_2, 1.0);
            cam.pan(yaw, pitch);
            let (fwd, rgt, up) = cam.basis();

            // All three must be unit-length.
            assert!(approx_eq(dot(fwd, fwd).sqrt(), 1.0),
                "fwd not unit at yaw={yaw} pitch={pitch}: {:.5}", dot(fwd, fwd).sqrt());
            assert!(approx_eq(dot(rgt, rgt).sqrt(), 1.0),
                "rgt not unit at yaw={yaw} pitch={pitch}");
            assert!(approx_eq(dot(up, up).sqrt(), 1.0),
                "up not unit at yaw={yaw} pitch={pitch}");

            // All pairs must be mutually perpendicular.
            assert!(approx_eq(dot(fwd, rgt), 0.0),
                "fwd · rgt = {:.5} at yaw={yaw} pitch={pitch}", dot(fwd, rgt));
            assert!(approx_eq(dot(fwd, up), 0.0),
                "fwd · up = {:.5} at yaw={yaw} pitch={pitch}", dot(fwd, up));
            assert!(approx_eq(dot(rgt, up), 0.0),
                "rgt · up = {:.5} at yaw={yaw} pitch={pitch}", dot(rgt, up));
        }
    }

    /// The right vector is defined as [cos(yaw), 0, sin(yaw)] — no Y component,
    /// ever. This guarantees that strafing is always horizontal regardless of
    /// how far the player is looking up or down (FPS camera convention).
    #[test]
    fn test_right_vector_has_no_y_component() {
        for pitch in [-1.0_f32, -0.5, 0.0, 0.5, 1.0] {
            let mut cam = Camera::new([0., 0., 0.], FRAC_PI_2, 1.0);
            cam.pan(0.0, pitch);
            let (_, rgt, _) = cam.basis();
            assert!(
                approx_eq(rgt[1], 0.0),
                "right vector Y should be 0 at pitch={pitch}, got {}", rgt[1],
            );
        }
    }

    /// CameraUniform must be exactly 64 bytes: four vec3+pad slots at 16 bytes
    /// each. If this fails, the WGSL uniform binding will silently read garbage.
    #[test]
    fn test_camera_uniform_layout() {
        assert_eq!(std::mem::size_of::<CameraUniform>(), 64);
    }

    /// RasterCameraUniform must be exactly 144 bytes: two 4×4 matrices (128 B)
    /// plus world-position vec3 + pad (16 B).
    #[test]
    fn test_raster_camera_uniform_layout() {
        assert_eq!(std::mem::size_of::<RasterCameraUniform>(), 144);
    }

    /// At the default orientation (yaw=0, pitch=0), the view matrix column 2
    /// should encode the −forward direction, which is +Z (since fwd = −Z).
    /// This confirms the view matrix is oriented correctly.
    #[test]
    fn test_raster_uniform_view_matrix_default() {
        let cam = Camera::new([0.0, 0.0, 0.0], FRAC_PI_2, 1.0);
        let u = cam.to_raster_uniform();
        // At default: fwd=(0,0,-1), rgt=(1,0,0), up=(0,1,0)
        // view col 2 (encodes -fwd direction): [-fwd.x, -fwd.y, -fwd.z, 0]
        //   = [0, 0, 1, 0]  (positive Z = "behind" camera = depth direction)
        assert!(approx_eq(u.view[2][0], 0.0), "view[2][0] should be 0, got {}", u.view[2][0]);
        assert!(approx_eq(u.view[2][1], 0.0), "view[2][1] should be 0, got {}", u.view[2][1]);
        assert!(approx_eq(u.view[2][2], 1.0), "view[2][2] should be 1, got {}", u.view[2][2]);
    }

    /// to_uniform() must produce a forward vector that matches what ray(0.5, 0.5)
    /// produces — both represent where the camera is pointing.
    #[test]
    fn test_to_uniform_fwd_matches_center_ray() {
        let cam = Camera::new([0.0, 0.0, 0.0], FRAC_PI_2, 1.0);
        let u = cam.to_uniform();
        let ray = cam.ray(0.5, 0.5);
        assert!(
            vec_approx_eq(u.fwd, ray.direction),
            "to_uniform fwd {:?} should match center ray direction {:?}",
            u.fwd, ray.direction,
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

    // ── Raster camera matrix math ─────────────────────────────────────────────

    /// Multiply a column-major 4×4 matrix by a homogeneous point [x, y, z, 1].
    /// Returns the transformed [x, y, z] (w division omitted — view matrices
    /// never scale w, so w is always 1.0 after a rigid-body view transform).
    fn view_transform(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
        let w = [p[0], p[1], p[2], 1.0f32];
        [
            m[0][0]*w[0] + m[1][0]*w[1] + m[2][0]*w[2] + m[3][0]*w[3],
            m[0][1]*w[0] + m[1][1]*w[1] + m[2][1]*w[2] + m[3][1]*w[3],
            m[0][2]*w[0] + m[1][2]*w[1] + m[2][2]*w[2] + m[3][2]*w[3],
        ]
    }

    /// Apply the projection matrix to a point on the camera Z axis (x=y=0) and
    /// return the resulting NDC z value (z_clip / w_clip).
    fn proj_ndc_z(proj: &[[f32; 4]; 4], cam_z: f32) -> f32 {
        // col-major: z_clip = proj[2][2]*cam_z + proj[3][2]
        //            w_clip = proj[2][3]*cam_z + proj[3][3]
        let z_clip = proj[2][2] * cam_z + proj[3][2];
        let w_clip = proj[2][3] * cam_z + proj[3][3];
        z_clip / w_clip
    }

    /// A camera at (0, 0, 5) looking toward the origin: the world origin should
    /// appear at camera-space z = −5 after applying the view matrix.
    ///
    /// This verifies that the view matrix correctly encodes the camera position
    /// and that world→camera-space transforms produce sensible depth values.
    #[test]
    fn test_raster_view_transforms_world_point() {
        let cam = Camera::new([0.0, 0.0, 5.0], FRAC_PI_2, 1.0);
        let u   = cam.to_raster_uniform();

        // World origin is 5 units in front of the camera (along −Z).
        // In camera space that maps to z = −5 (negative = in front).
        let cam_space = view_transform(&u.view, [0.0, 0.0, 0.0]);
        assert!(approx_eq(cam_space[0],  0.0), "x should be 0, got {}", cam_space[0]);
        assert!(approx_eq(cam_space[1],  0.0), "y should be 0, got {}", cam_space[1]);
        assert!(approx_eq(cam_space[2], -5.0), "z should be −5, got {}", cam_space[2]);
    }

    /// A camera at (3, 1, 0), looking along −Z (default orientation).
    /// A point directly ahead at (3, 1, −7) should transform to (0, 0, −7)
    /// in camera space — the lateral offsets cancel out.
    #[test]
    fn test_raster_view_offsets_cancel() {
        let cam = Camera::new([3.0, 1.0, 0.0], FRAC_PI_2, 1.0);
        let u   = cam.to_raster_uniform();

        let cam_space = view_transform(&u.view, [3.0, 1.0, -7.0]);
        assert!(approx_eq(cam_space[0],  0.0), "x should cancel to 0, got {}", cam_space[0]);
        assert!(approx_eq(cam_space[1],  0.0), "y should cancel to 0, got {}", cam_space[1]);
        assert!(approx_eq(cam_space[2], -7.0), "z should be −7, got {}",       cam_space[2]);
    }

    /// WebGPU NDC depth range is [0, 1] (not OpenGL's [−1, 1]).
    /// The near plane must map to NDC z = 0 and the far plane to NDC z = 1.
    ///
    /// If this fails the depth buffer will have the wrong range and distant
    /// objects will be clipped or z-fight with near ones.
    #[test]
    fn test_raster_proj_near_far_depth_mapping() {
        let cam  = Camera::new([0.0, 0.0, 0.0], FRAC_PI_2, 1.0);
        let u    = cam.to_raster_uniform();
        let near = 0.01_f32;
        let far  = 1000.0_f32;

        let ndc_near = proj_ndc_z(&u.proj, -near);
        let ndc_far  = proj_ndc_z(&u.proj, -far);

        assert!(approx_eq(ndc_near, 0.0), "near plane → NDC z=0, got {ndc_near}");
        assert!(approx_eq(ndc_far,  1.0), "far plane  → NDC z=1, got {ndc_far}");
    }

    /// The projection matrix must encode the viewport aspect ratio correctly:
    /// the ratio of vertical to horizontal scale equals the aspect ratio.
    /// A wrong aspect ratio stretches or squishes the rendered geometry.
    #[test]
    fn test_raster_proj_encodes_aspect_ratio() {
        let aspect = 16.0_f32 / 9.0;
        let cam    = Camera::new([0.0, 0.0, 0.0], FRAC_PI_2, aspect);
        let u      = cam.to_raster_uniform();

        // proj[1][1] = 1/tan(vfov/2)
        // proj[0][0] = 1/(aspect × tan(vfov/2))
        // Ratio should equal aspect.
        let ratio = u.proj[1][1] / u.proj[0][0];
        assert!(
            (ratio - aspect).abs() < 1e-4,
            "proj[1][1]/proj[0][0] should equal aspect {aspect}, got {ratio}"
        );
    }

    /// The `cam_pos` field of `RasterCameraUniform` must equal the camera's
    /// world-space position so the fragment shader can compute view directions.
    #[test]
    fn test_raster_cam_pos_passthrough() {
        let pos = [4.2_f32, -1.1, 3.3];
        let cam = Camera::new(pos, FRAC_PI_2, 1.0);
        let u   = cam.to_raster_uniform();
        assert!(approx_eq(u.cam_pos[0], pos[0]), "cam_pos.x");
        assert!(approx_eq(u.cam_pos[1], pos[1]), "cam_pos.y");
        assert!(approx_eq(u.cam_pos[2], pos[2]), "cam_pos.z");
    }
}
