/**
 * Turntable camera controller.
 *
 * Models the camera as a point on the surface of a sphere. The sphere's
 * centre is `target` (the scene's world-space centre); the camera sits on
 * the surface at (azimuth, elevation) and looks inward. Dragging pans the
 * viewpoint around the sphere; scrolling changes the radius.
 *
 * This produces pixels identical to spinning the object in front of a fixed
 * lens — the GPU scene geometry never moves.
 *
 * Angles:
 *   azimuth   — horizontal rotation in radians (0 = camera at +Z, looking −Z)
 *   elevation — vertical tilt in radians (positive = camera above the equator)
 *
 * Drag convention:
 *   drag right → azimuth decreases → right face of object becomes visible
 *   drag up    → elevation increases → top of object tilts toward you
 */
export class Turntable {
    /** World-space point the camera orbits around. */
    target: [number, number, number] = [0, 0, 0];

    /** Horizontal orbit angle in radians. */
    azimuth = 0;

    /** Vertical orbit angle in radians. Positive = above the equator. */
    elevation = 0.25;  // ~14° above horizontal — slightly elevated start looks best

    /** Distance from target to camera position. */
    radius = 5;

    /**
     * Maximum elevation angle. Clamped to just below ±90° so the camera
     * never flips upside-down, which would invert the drag direction and
     * confuse everyone involved.
     */
    static readonly EL_MAX = Math.PI / 2 - 0.05;

    /**
     * Pan the viewpoint by angular deltas in radians.
     *
     * `dAzimuth` and `dElevation` are already in radians — the caller is
     * responsible for multiplying pixel deltas by a drag-speed constant.
     * Elevation is clamped to ±EL_MAX.
     */
    pan(dAzimuth: number, dElevation: number): void {
        this.azimuth  += dAzimuth;
        this.elevation = Math.max(
            -Turntable.EL_MAX,
            Math.min(Turntable.EL_MAX, this.elevation + dElevation),
        );
    }

    /**
     * Scale the orbit radius by `factor`.
     *
     * Use `Math.exp(wheelDelta * speed)` for exponential zoom: equal up/down
     * scrolls cancel perfectly and the zoom feels proportional at any distance.
     * `minRadius` prevents the camera from clipping inside the model.
     */
    zoom(factor: number, minRadius = 0.01): void {
        this.radius = Math.max(minRadius, this.radius * factor);
    }

    /**
     * Reframe the camera so the given scene bounding box fills the viewport.
     *
     * Sets the orbit target to the box centre and the radius to the full
     * diagonal length, which guarantees the model is always fully visible
     * regardless of orientation. Resets azimuth and elevation to the default
     * starting view.
     *
     * `bounds` must be `[minX, minY, minZ, maxX, maxY, maxZ]` — the format
     * returned by the WASM `get_scene_bounds()` call.
     */
    autoFrame(bounds: Float32Array): void {
        if (bounds.length < 6) return;
        this.target = [
            (bounds[0] + bounds[3]) / 2,
            (bounds[1] + bounds[4]) / 2,
            (bounds[2] + bounds[5]) / 2,
        ];
        this.azimuth   = 0;
        this.elevation = 0.25;
        const dx = bounds[3] - bounds[0];
        const dy = bounds[4] - bounds[1];
        const dz = bounds[5] - bounds[2];
        this.radius = Math.max(Math.sqrt(dx * dx + dy * dy + dz * dz), 0.1);
    }

    /**
     * Returns the camera position and orientation parameters expected by the
     * WASM `update_camera()` call. Pure math — no side effects.
     *
     * Maps spherical coordinates to Cartesian position plus yaw/pitch:
     *
     *   position = target + r*(sin(az)*cos(el), sin(el), cos(az)*cos(el))
     *   yaw      = −az    (camera looks toward target, not away from it)
     *   pitch    = −el
     *
     * The sign flip arises from the Rust camera's forward-vector formula:
     *   fwd = (sin(yaw)*cos(pitch), sin(pitch), −cos(yaw)*cos(pitch))
     * Setting yaw=−az, pitch=−el makes fwd point from position toward target.
     */
    toCameraParams(): { px: number; py: number; pz: number; yaw: number; pitch: number } {
        const { target: [tx, ty, tz], azimuth: az, elevation: el, radius: r } = this;
        const cosEl = Math.cos(el);
        return {
            px:    tx + r * Math.sin(az) * cosEl,
            py:    ty + r * Math.sin(el),
            pz:    tz + r * Math.cos(az) * cosEl,
            yaw:   -az,
            pitch: -el,
        };
    }
}
