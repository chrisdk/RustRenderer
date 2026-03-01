/**
 * Render state machine — the async orchestration layer between the UI and the GPU.
 *
 * This module owns all the boolean flags that govern what the render loop is
 * allowed to do at any moment, and the async functions that modify those flags.
 * Isolating state here makes the transitions testable without a browser, DOM,
 * or real WebGPU device: inject fake deps, set flags by hand, advance fake
 * timers, and assert.
 *
 * Two invariants that must never be violated:
 *   1. Only one GPU operation runs at a time. Both `rendering` and `hqRendering`
 *      must be false before scene data can be replaced.
 *   2. `hqCancelled` is a soft stop: it signals the HQ loop to exit after its
 *      current sample, not to abort mid-frame.
 */
/** Milliseconds between polls when waiting for the GPU pipeline to drain. */
const POLL_MS = 16;
function sleep() {
    return new Promise(resolve => setTimeout(resolve, POLL_MS));
}
/**
 * Render loop state machine.
 *
 * Owns the boolean flags that gate what the render loop may do, and the async
 * methods that drive state transitions. All GPU-mutating calls are routed
 * through the `deps` interface so the class is testable without a browser or
 * real WebGPU device.
 */
export class RenderController {
    /**
     * @param deps       External dependencies (WASM, canvas, UI callbacks).
     * @param hqSamples  Samples per pixel for a full-quality render.
     *                   Overridable so tests can avoid running 256 iterations.
     *                   Mutable so the sample-count UI control can update it
     *                   between renders without constructing a new controller.
     */
    constructor(deps, hqSamples = 256) {
        this.deps = deps;
        this.hqSamples = hqSamples;
        /** True once a scene has been successfully loaded. */
        this.sceneLoaded = false;
        /**
         * True when the camera has moved since the last raster frame.
         * Set by callers (drag, zoom, resize, scene load); cleared by tick().
         */
        this.cameraDirty = false;
        /**
         * True while a single async GPU operation is in flight.
         * Both the rasterizer and path tracer set this for the duration of their
         * staging-buffer readback. Prevents concurrent GPU submissions.
         */
        this.rendering = false;
        /**
         * True for the lifetime of the HQ progressive render loop, from the first
         * sample dispatch until the loop exits (cleanly or cancelled).
         */
        this.hqRendering = false;
        /**
         * Soft-stop flag. Set to true to ask the HQ loop to exit after its current
         * sample. Cleared at the start of every new HQ render so stale
         * cancellations never carry over.
         */
        this.hqCancelled = false;
        /**
         * Stable identifier for the currently loaded scene.
         * A new load with a different key cancels any in-progress HQ render.
         * A new load with the same key leaves it running.
         */
        this.currentSceneKey = null;
    }
    // ── Scene loading ──────────────────────────────────────────────────────────
    /**
     * Replace the current scene with new GLTF/GLB bytes.
     *
     * `key` is a stable identifier for the scene (a filename or built-in name).
     * If it differs from the currently loaded scene and an HQ render is running,
     * the render is cancelled first. Either way we wait for the GPU pipeline to
     * go completely idle before handing new scene data to WASM — calling
     * `load_scene` while a staging-buffer readback is in flight crashes the
     * WebGPU layer (getMappedRange on a null buffer).
     */
    async loadSceneBytes(bytes, key) {
        // Cancel the HQ render if we're switching to a different scene.
        // Same-scene reloads leave it running — the render is still valid.
        if (key !== this.currentSceneKey && this.hqRendering) {
            this.hqCancelled = true;
        }
        // Drain the GPU: wait for both the HQ loop and any in-flight raster
        // frame to finish. A raster frame can start in the window after
        // hqRendering goes false but before we reach load_scene (tick() fires
        // in that gap), so both flags must be checked.
        while (this.hqRendering || this.rendering) {
            await sleep();
        }
        this.deps.setStatus('Loading scene…');
        try {
            this.deps.load_scene(bytes);
            this.currentSceneKey = key;
            this.deps.onSceneLoaded(this.deps.get_scene_bounds());
            this.sceneLoaded = true;
            this.cameraDirty = true;
            this.deps.setStatus('Drag to spin · click Render for full quality');
        }
        catch (err) {
            this.deps.setStatus(`Error: ${err}`);
        }
    }
    // ── High-quality progressive render ───────────────────────────────────────
    /**
     * Accumulate `hqSamples` path-traced samples into a single converging image.
     *
     * Each call to `deps.render` adds one independent set of random light paths;
     * they are averaged in WASM. The result is written to the canvas after every
     * sample so the image visibly improves while the render runs.
     */
    async startHighQualityRender() {
        if (!this.sceneLoaded || this.hqRendering)
            return;
        // Wait for any in-flight raster frame before claiming `rendering`.
        while (this.rendering) {
            await sleep();
        }
        this.hqRendering = true;
        this.hqCancelled = false;
        this.rendering = true;
        this.deps.onHqStart();
        const { w, h } = this.deps.getCanvasSize();
        this.deps.lockCamera(w, h);
        for (let i = 0; i < this.hqSamples && !this.hqCancelled; i++) {
            this.deps.render(w, h, i, false);
            const pixels = await this.deps.get_pixels(w, h);
            this.deps.putImageData(pixels, w, h);
            this.deps.setStatus(`Rendering… ${i + 1} / ${this.hqSamples} spp`);
        }
        const wasCancelled = this.hqCancelled;
        this.rendering = false;
        this.hqRendering = false;
        this.deps.onHqEnd(wasCancelled);
        if (wasCancelled) {
            this.deps.setStatus('Render cancelled — drag to spin');
            this.cameraDirty = true;
        }
        else {
            this.deps.setStatus(`Done — ${this.hqSamples} spp`);
        }
    }
    /** Signal the running HQ render to stop after its current sample. No-op if idle. */
    cancelHighQualityRender() {
        if (this.hqRendering) {
            this.hqCancelled = true;
        }
    }
    // ── Raster preview frame ───────────────────────────────────────────────────
    /**
     * Dispatch one hardware-rasterized preview frame.
     *
     * Returns immediately if a render is already in flight — tick() calls this
     * every animation frame, so the re-entrancy guard is essential.
     */
    async renderDragFrame() {
        if (this.rendering)
            return;
        this.rendering = true;
        const { w, h } = this.deps.getCanvasSize();
        this.deps.raster_frame(w, h);
        const pixels = await this.deps.raster_get_pixels(w, h);
        this.deps.putImageData(pixels, w, h);
        this.rendering = false;
    }
    // ── Tick guard ─────────────────────────────────────────────────────────────
    /**
     * Returns true when the animation loop should dispatch a raster preview frame.
     * All four conditions must hold:
     *   - A scene is loaded (nothing to draw without one)
     *   - The camera has moved since the last draw (avoid wasted GPU work)
     *   - No GPU operation is in flight (one submission at a time)
     *   - The HQ render is not running (it owns the canvas while active)
     */
    shouldRenderDragFrame() {
        return !this.hqRendering && this.sceneLoaded && !this.rendering && this.cameraDirty;
    }
}
