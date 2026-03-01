/**
 * Unit tests for RenderController — the async state machine that governs
 * what the GPU render loop is allowed to do at any moment.
 *
 * Tests that involve the polling loop (while hqRendering || rendering) use
 * fake timers so they run instantly rather than waiting real wall-clock time.
 * Tests for purely synchronous behaviour use plain assertions.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { RenderController } from './render-controller.js';
// ── Test helpers ──────────────────────────────────────────────────────────────
/** Build a fully-mocked RenderDeps. Every method is a vi.fn(). */
function makeDeps() {
    return {
        load_scene: vi.fn(),
        get_scene_bounds: vi.fn(() => new Float32Array([0, 0, 0, 1, 1, 1])),
        render: vi.fn(),
        get_pixels: vi.fn(async () => new Uint8Array(4)),
        raster_frame: vi.fn(),
        raster_get_pixels: vi.fn(async () => new Uint8Array(4)),
        putImageData: vi.fn(),
        getCanvasSize: vi.fn(() => ({ w: 100, h: 100 })),
        setStatus: vi.fn(),
        onSceneLoaded: vi.fn(),
        lockCamera: vi.fn(),
        onHqStart: vi.fn(),
        onHqEnd: vi.fn(),
    };
}
/**
 * Create a controller + deps pair. hqSamples defaults to 4 so the full render
 * loop terminates quickly in tests; pass a larger value when testing sample counts.
 */
function make(hqSamples = 4) {
    const deps = makeDeps();
    const ctrl = new RenderController(deps, hqSamples);
    return { ctrl, deps };
}
// ── loadSceneBytes ─────────────────────────────────────────────────────────────
describe('loadSceneBytes', () => {
    describe('when the GPU is idle', () => {
        it('calls load_scene, sets currentSceneKey, sceneLoaded, and cameraDirty', async () => {
            const { ctrl, deps } = make();
            await ctrl.loadSceneBytes(new Uint8Array(), 'scene-a');
            expect(deps.load_scene).toHaveBeenCalledOnce();
            expect(ctrl.currentSceneKey).toBe('scene-a');
            expect(ctrl.sceneLoaded).toBe(true);
            expect(ctrl.cameraDirty).toBe(true);
        });
        it('reports an error and does not update state when load_scene throws', async () => {
            const { ctrl, deps } = make();
            deps.load_scene = vi.fn(() => { throw new Error('corrupt glb'); });
            await ctrl.loadSceneBytes(new Uint8Array(), 'scene-a');
            expect(ctrl.currentSceneKey).toBeNull();
            expect(ctrl.sceneLoaded).toBe(false);
            expect(deps.setStatus).toHaveBeenCalledWith(expect.stringContaining('Error'));
        });
    });
    // The next four tests exercise the GPU-drain loop. Fake timers let us
    // control when the polling setTimeout fires without real wall-clock waits.
    describe('waiting for the GPU to drain', () => {
        beforeEach(() => vi.useFakeTimers());
        afterEach(() => vi.useRealTimers());
        it('does not call load_scene while rendering is true (raster-frame-in-flight bug)', async () => {
            const { ctrl, deps } = make();
            ctrl.rendering = true;
            const p = ctrl.loadSceneBytes(new Uint8Array(), 'scene-a');
            // One poll tick — still blocked.
            await vi.advanceTimersByTimeAsync(16);
            expect(deps.load_scene).not.toHaveBeenCalled();
            // Unblock, advance again — now load_scene should fire.
            ctrl.rendering = false;
            await vi.advanceTimersByTimeAsync(16);
            await p;
            expect(deps.load_scene).toHaveBeenCalledOnce();
        });
        it('does not call load_scene while hqRendering is true', async () => {
            const { ctrl, deps } = make();
            ctrl.hqRendering = true;
            const p = ctrl.loadSceneBytes(new Uint8Array(), 'scene-b');
            await vi.advanceTimersByTimeAsync(16);
            expect(deps.load_scene).not.toHaveBeenCalled();
            ctrl.hqRendering = false;
            await vi.advanceTimersByTimeAsync(16);
            await p;
            expect(deps.load_scene).toHaveBeenCalledOnce();
        });
        it('sets hqCancelled when switching to a different scene', async () => {
            const { ctrl } = make();
            ctrl.hqRendering = true;
            ctrl.currentSceneKey = 'scene-a';
            const p = ctrl.loadSceneBytes(new Uint8Array(), 'scene-b');
            // The cancel flag is set synchronously before any await.
            expect(ctrl.hqCancelled).toBe(true);
            ctrl.hqRendering = false;
            await vi.advanceTimersByTimeAsync(16);
            await p;
        });
        it('does not set hqCancelled when the same scene is re-selected', async () => {
            const { ctrl } = make();
            ctrl.hqRendering = true;
            ctrl.currentSceneKey = 'scene-a';
            const p = ctrl.loadSceneBytes(new Uint8Array(), 'scene-a');
            expect(ctrl.hqCancelled).toBe(false);
            ctrl.hqRendering = false;
            await vi.advanceTimersByTimeAsync(16);
            await p;
        });
    });
});
// ── startHighQualityRender ─────────────────────────────────────────────────────
describe('startHighQualityRender', () => {
    it('returns immediately when no scene is loaded', async () => {
        const { ctrl, deps } = make();
        await ctrl.startHighQualityRender();
        expect(deps.render).not.toHaveBeenCalled();
    });
    it('returns immediately when an HQ render is already running', async () => {
        const { ctrl, deps } = make();
        ctrl.sceneLoaded = true;
        ctrl.hqRendering = true;
        await ctrl.startHighQualityRender();
        expect(deps.render).not.toHaveBeenCalled();
    });
    it('calls lockCamera once per render run', async () => {
        const { ctrl, deps } = make(1);
        ctrl.sceneLoaded = true;
        await ctrl.startHighQualityRender();
        expect(deps.lockCamera).toHaveBeenCalledOnce();
    });
    it('dispatches exactly hqSamples render calls then clears all flags', async () => {
        const { ctrl, deps } = make(4);
        ctrl.sceneLoaded = true;
        await ctrl.startHighQualityRender();
        expect(deps.render).toHaveBeenCalledTimes(4);
        expect(ctrl.rendering).toBe(false);
        expect(ctrl.hqRendering).toBe(false);
    });
    it('stops after the in-progress sample when hqCancelled is set mid-run', async () => {
        const { ctrl, deps } = make(10);
        ctrl.sceneLoaded = true;
        let callCount = 0;
        deps.get_pixels = vi.fn(async () => {
            // Cancel after the second sample completes.
            if (++callCount === 2)
                ctrl.hqCancelled = true;
            return new Uint8Array(4);
        });
        await ctrl.startHighQualityRender();
        // Loop exits at the top of iteration 3 when it sees hqCancelled.
        expect(deps.render).toHaveBeenCalledTimes(2);
        expect(ctrl.cameraDirty).toBe(true);
        expect(ctrl.rendering).toBe(false);
        expect(ctrl.hqRendering).toBe(false);
    });
    it('sets cameraDirty on cancel, leaves it false on normal completion', async () => {
        // Normal completion — cameraDirty should stay false.
        const { ctrl: ctrlA } = make(2);
        ctrlA.sceneLoaded = true;
        await ctrlA.startHighQualityRender();
        expect(ctrlA.cameraDirty).toBe(false);
        // Cancelled — cameraDirty must be set so the raster loop redraws.
        const { ctrl: ctrlB, deps: depsB } = make(10);
        ctrlB.sceneLoaded = true;
        depsB.get_pixels = vi.fn(async () => {
            ctrlB.hqCancelled = true;
            return new Uint8Array(4);
        });
        await ctrlB.startHighQualityRender();
        expect(ctrlB.cameraDirty).toBe(true);
    });
});
// ── renderDragFrame ────────────────────────────────────────────────────────────
describe('renderDragFrame', () => {
    it('returns immediately when rendering is already true', async () => {
        const { ctrl, deps } = make();
        ctrl.rendering = true;
        await ctrl.renderDragFrame();
        expect(deps.raster_frame).not.toHaveBeenCalled();
    });
    it('calls raster_frame, raster_get_pixels, putImageData, then clears rendering', async () => {
        const { ctrl, deps } = make();
        await ctrl.renderDragFrame();
        expect(deps.raster_frame).toHaveBeenCalledOnce();
        expect(deps.raster_get_pixels).toHaveBeenCalledOnce();
        expect(deps.putImageData).toHaveBeenCalledOnce();
        expect(ctrl.rendering).toBe(false);
    });
});
// ── shouldRenderDragFrame ──────────────────────────────────────────────────────
describe('shouldRenderDragFrame', () => {
    it('returns true when all conditions are met', () => {
        const { ctrl } = make();
        ctrl.sceneLoaded = true;
        ctrl.cameraDirty = true;
        expect(ctrl.shouldRenderDragFrame()).toBe(true);
    });
    it('returns false when hqRendering is true', () => {
        const { ctrl } = make();
        ctrl.sceneLoaded = true;
        ctrl.cameraDirty = true;
        ctrl.hqRendering = true;
        expect(ctrl.shouldRenderDragFrame()).toBe(false);
    });
    it('returns false when rendering is true', () => {
        const { ctrl } = make();
        ctrl.sceneLoaded = true;
        ctrl.cameraDirty = true;
        ctrl.rendering = true;
        expect(ctrl.shouldRenderDragFrame()).toBe(false);
    });
    it('returns false when cameraDirty is false', () => {
        const { ctrl } = make();
        ctrl.sceneLoaded = true;
        // cameraDirty defaults to false
        expect(ctrl.shouldRenderDragFrame()).toBe(false);
    });
    it('returns false when no scene is loaded', () => {
        const { ctrl } = make();
        ctrl.cameraDirty = true;
        // sceneLoaded defaults to false
        expect(ctrl.shouldRenderDragFrame()).toBe(false);
    });
});
