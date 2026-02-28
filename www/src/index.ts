/**
 * Path tracer frontend.
 *
 * Responsibilities:
 *   1. Load the WASM module and initialise the GPU renderer.
 *   2. Accept a GLTF/GLB scene via drag-and-drop or file picker.
 *   3. Turntable camera: click-drag spins the scene in front of a fixed camera.
 *   4. Preview: update_camera → raster_frame → raster_get_pixels (60 fps rasterizer).
 *   5. HQ render: update_camera → render → get_pixels → putImageData (path tracer).
 */

import init, {
    init_renderer,
    load_scene,
    load_environment,
    update_camera,
    render,
    get_pixels,
    raster_frame,
    raster_get_pixels,
    get_scene_bounds,
} from '../pkg/render.js';
import { BUILTIN_SCENES } from './scenes.js';

// ─────────────────────────────────────────────────────────────────────────────
// DOM references
// ─────────────────────────────────────────────────────────────────────────────

const canvas      = document.getElementById('canvas')        as HTMLCanvasElement;
const ctx         = canvas.getContext('2d')!;
const statusEl    = document.getElementById('status')!;
const fileInput   = document.getElementById('file-input')    as HTMLInputElement;
const envInput    = document.getElementById('env-file-input') as HTMLInputElement;
const hintsEl     = document.getElementById('hints')!;
const dropOverlay = document.getElementById('drop-overlay')!;
const scenePicker = document.getElementById('scene-picker')!;
const renderBtn   = document.getElementById('render-btn')    as HTMLButtonElement;

// ─────────────────────────────────────────────────────────────────────────────
// Turntable camera state
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Turntable camera: the scene rotates in front of a stationary camera.
 *
 * Mechanically we move the camera on a sphere whose centre is `target` (the
 * scene's world-space centre) and whose radius is `radius`. This produces
 * pixels identical to spinning the object itself, at a fraction of the
 * complexity — the GPU scene geometry never moves.
 *
 * Angles:
 *   azimuth  — horizontal rotation in radians (0 = camera at +Z, looking −Z)
 *   elevation — vertical tilt in radians (positive = camera above the equator)
 *
 * Drag convention (turntable, not "fly-around"):
 *   drag right → azimuth increases → right face of object becomes visible
 *   drag up    → elevation increases → top of object tilts toward you
 */
const turntable = {
    target:    [0, 0, 0] as [number, number, number],
    azimuth:   0,
    elevation: 0.25,   // ~14° above horizontal — slightly elevated start looks best
    radius:    5,
};

const VFOV      = Math.PI / 3;           // 60° vertical field of view
const DRAG_SPEED = 0.005;               // radians of rotation per pixel dragged
const EL_MAX    = Math.PI / 2 - 0.05;   // elevation clamp — prevents camera flip

/**
 * Push the current turntable state to the GPU.
 *
 * Converts spherical (azimuth, elevation, radius) to Cartesian camera position
 * plus the yaw/pitch angles the WASM API expects. The mapping is:
 *
 *   position = target + r * (sin(az)*cos(el),  sin(el),  cos(az)*cos(el))
 *   yaw      = −az      (camera looks toward target, not away from it)
 *   pitch    = −el
 *
 * The sign flip comes from the FPS forward-vector formula:
 *   fwd = (sin(yaw)*cos(pitch), sin(pitch), −cos(yaw)*cos(pitch))
 * Setting yaw=−az, pitch=−el makes fwd point from position toward target.
 */
function applyTurntable(): void {
    const { target: [tx, ty, tz], azimuth: az, elevation: el, radius: r } = turntable;

    const cosEl = Math.cos(el);
    const px = tx + r * Math.sin(az) * cosEl;
    const py = ty + r * Math.sin(el);
    const pz = tz + r * Math.cos(az) * cosEl;

    const w = canvas.width  || 1;
    const h = canvas.height || 1;

    update_camera(px, py, pz, -az, -el, VFOV, w / h);
    cameraDirty = true;
}

/**
 * Position the camera so the loaded scene fills roughly 80% of the viewport
 * in its largest on-screen dimension.
 *
 * The radius is set to 75% of the bounding-box diagonal, which equals 1.5×
 * the scene's circumscribed sphere radius — so the camera is always safely
 * outside the model regardless of orientation.
 */
function applyAutoCamera(): void {
    const b = get_scene_bounds();
    if (b.length < 6) return;

    turntable.target    = [(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2];
    turntable.azimuth   = 0;
    turntable.elevation = 0.25;

    const dx = b[3] - b[0], dy = b[4] - b[1], dz = b[5] - b[2];
    turntable.radius = Math.max(Math.sqrt(dx * dx + dy * dy + dz * dz) * 1.0, 0.1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Drag input
// ─────────────────────────────────────────────────────────────────────────────

let dragging     = false;
let lastX        = 0;
let lastY        = 0;

canvas.addEventListener('pointerdown', e => {
    if (!sceneLoaded) return;
    canvas.setPointerCapture(e.pointerId);
    dragging = true;
    lastX    = e.clientX;
    lastY    = e.clientY;
    canvas.classList.add('dragging');
    if (hqRendering) hqCancelled = true;
});

canvas.addEventListener('pointermove', e => {
    if (!dragging) return;

    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    turntable.azimuth   -= dx * DRAG_SPEED;
    turntable.elevation  = Math.max(-EL_MAX,
                           Math.min( EL_MAX,
                           turntable.elevation + dy * DRAG_SPEED));

    applyTurntable();
});

canvas.addEventListener('pointerup',     () => endDrag());
canvas.addEventListener('pointercancel', () => endDrag());

function endDrag(): void {
    dragging = false;
    canvas.classList.remove('dragging');
}

// ─────────────────────────────────────────────────────────────────────────────
// Render loop state  (declared before resizeCanvas so the TDZ doesn't bite)
// ─────────────────────────────────────────────────────────────────────────────

let sceneLoaded = false;
let cameraDirty = false;
let rendering   = false;   // true while an async get_pixels call is in flight

// High-quality progressive render state.
let hqRendering = false;   // true while the multi-sample render loop is running
let hqCancelled = false;   // set to true to stop the loop after the current sample

const HQ_SAMPLES = 256;    // samples per pixel for a full quality render

// ── Display strategy ──────────────────────────────────────────────────────────
//
// The rasterizer is the primary display: it renders whenever the camera changes
// (drag, resize, scene load) and its output stays on screen until something
// replaces it. The path tracer is only used for the explicit "Render" button.
//
// This means the canvas always shows a sharp, textured, correctly-lit image
// even when idle — no progressive noise build-up before the user renders.

// ─────────────────────────────────────────────────────────────────────────────
// Canvas sizing
// ─────────────────────────────────────────────────────────────────────────────

function resizeCanvas(): void {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width  = w;
        canvas.height = h;
        cameraDirty   = true;
    }
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// ─────────────────────────────────────────────────────────────────────────────
// Render helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Render one preview frame using the hardware rasterizer.
 *
 * The rasterizer runs entirely on the GPU (no ray traversal, no bounce
 * accumulation) and produces a single complete frame at full canvas resolution
 * in a fraction of a millisecond of GPU time. Called while the camera is in
 * motion so the turntable stays snappy even on complex scenes.
 *
 * The camera is already uploaded before this is called (applyTurntable() does
 * it), so we just dispatch and blit.
 */
async function renderDragFrame(): Promise<void> {
    if (rendering) return;
    rendering = true;

    const w = canvas.width;
    const h = canvas.height;

    raster_frame(w, h);
    const pixels = await raster_get_pixels(w, h);
    ctx.putImageData(new ImageData(new Uint8ClampedArray(pixels), w, h), 0, 0);

    rendering = false;
}

/**
 * Main animation loop.
 *
 * Renders a raster frame whenever the camera changes (drag, resize, scene
 * load). The rasterizer output stays on screen until the HQ path tracer
 * explicitly replaces it — so the canvas is always showing something useful.
 */
function tick(): void {
    if (!hqRendering && sceneLoaded && !rendering && cameraDirty) {
        cameraDirty = false;
        renderDragFrame();
    }

    requestAnimationFrame(tick);
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene loading
// ─────────────────────────────────────────────────────────────────────────────

async function loadSceneBytes(bytes: Uint8Array): Promise<void> {
    setStatus('Loading scene…');
    try {
        load_scene(bytes);
        applyAutoCamera();
        applyTurntable();
        sceneLoaded = true;
        renderBtn.disabled = false;
        setStatus('Drag to spin · click Render for full quality');
        fadeHints();
    } catch (err) {
        setStatus(`Error: ${err}`);
    }
}

/**
 * Decodes and uploads a Radiance HDR (.hdr) environment map.
 *
 * Once loaded, the path tracer uses it for background, ambient, and
 * reflections instead of the procedural sky gradient. The rasterizer
 * preview is unaffected — it continues using its baked sky colours.
 */
async function loadEnvironmentBytes(bytes: Uint8Array): Promise<void> {
    setStatus('Loading environment…');
    try {
        load_environment(bytes);
        setStatus('Environment loaded · click Render to apply IBL');
    } catch (err) {
        setStatus(`Environment error: ${err}`);
    }
}

// ── High-quality render ───────────────────────────────────────────────────────

/**
 * Progressively accumulate HQ_SAMPLES samples into the same image.
 */
async function startHighQualityRender(): Promise<void> {
    if (!sceneLoaded || hqRendering) return;

    while (rendering) {
        await new Promise<void>(r => setTimeout(r, 16));
    }

    hqRendering = true;
    hqCancelled = false;
    rendering   = true;
    renderBtn.textContent = 'Cancel';
    renderBtn.classList.add('cancel');

    const w = canvas.width;
    const h = canvas.height;

    // Lock in the current camera for the full render.
    const { target: [tx, ty, tz], azimuth: az, elevation: el, radius: r } = turntable;
    const cosEl = Math.cos(el);
    update_camera(
        tx + r * Math.sin(az) * cosEl,
        ty + r * Math.sin(el),
        tz + r * Math.cos(az) * cosEl,
        -az, -el, VFOV, w / h,
    );

    for (let i = 0; i < HQ_SAMPLES && !hqCancelled; i++) {
        render(w, h, i, false);
        const pixels = await get_pixels(w, h);
        ctx.putImageData(new ImageData(new Uint8ClampedArray(pixels), w, h), 0, 0);
        setStatus(`Rendering… ${i + 1} / ${HQ_SAMPLES} spp`);
    }

    rendering   = false;
    hqRendering = false;
    renderBtn.textContent = 'Render';
    renderBtn.classList.remove('cancel');

    if (hqCancelled) {
        setStatus('Render cancelled — drag to spin');
        cameraDirty = true;
    } else {
        setStatus(`Done — ${HQ_SAMPLES} spp`);
    }
}

renderBtn.addEventListener('click', () => {
    if (hqRendering) {
        hqCancelled = true;
    } else {
        startHighQualityRender();
    }
});

// ── Built-in scene picker ─────────────────────────────────────────────────────

function buildScenePicker(): void {
    let activeBtn: HTMLButtonElement | null = null;

    for (const scene of BUILTIN_SCENES) {
        const btn = document.createElement('button');
        btn.className   = 'scene-btn';
        btn.textContent = scene.label;

        btn.addEventListener('click', async () => {
            if (activeBtn) activeBtn.classList.remove('active');
            activeBtn = btn;
            btn.classList.add('active');

            setStatus('Downloading scene…');
            const glb = await Promise.resolve(scene.build());
            await loadSceneBytes(new Uint8Array(glb));
        });

        scenePicker.appendChild(btn);

        // Show required attribution below the button for non-CC0 assets.
        if (scene.attribution) {
            const attr = document.createElement('div');
            attr.className   = 'scene-attribution';
            attr.textContent = scene.attribution;
            scenePicker.appendChild(attr);
        }
    }
}

// ── File picker ──────────────────────────────────────────────────────────────

fileInput.addEventListener('change', async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    const buffer = await file.arrayBuffer();
    await loadSceneBytes(new Uint8Array(buffer));
});

envInput.addEventListener('change', async () => {
    const file = envInput.files?.[0];
    if (!file) return;
    const buffer = await file.arrayBuffer();
    await loadEnvironmentBytes(new Uint8Array(buffer));
});

// ── Drag-and-drop ────────────────────────────────────────────────────────────

window.addEventListener('dragenter', e => {
    e.preventDefault();
    dropOverlay.classList.add('active');
});
window.addEventListener('dragover', e => {
    e.preventDefault();
});
window.addEventListener('dragleave', e => {
    if ((e as DragEvent).relatedTarget === null) {
        dropOverlay.classList.remove('active');
    }
});
window.addEventListener('drop', async e => {
    e.preventDefault();
    dropOverlay.classList.remove('active');
    const file = e.dataTransfer?.files[0];
    if (!file) return;
    const buffer = await file.arrayBuffer();
    // Route by file extension: .hdr → environment map; anything else → scene.
    if (file.name.toLowerCase().endsWith('.hdr')) {
        await loadEnvironmentBytes(new Uint8Array(buffer));
    } else {
        await loadSceneBytes(new Uint8Array(buffer));
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// UI helpers
// ─────────────────────────────────────────────────────────────────────────────

function setStatus(msg: string): void {
    statusEl.textContent = msg;
}

function fadeHints(): void {
    setTimeout(() => hintsEl.classList.add('fade'), 4000);
}

// ─────────────────────────────────────────────────────────────────────────────
// Boot sequence
// ─────────────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
    setStatus('Loading WASM…');
    try {
        await init();
    } catch (err) {
        setStatus(`WASM load failed: ${err}`);
        console.error('WASM init failed:', err);
        return;
    }

    buildScenePicker();

    setStatus('Requesting GPU…');
    try {
        await init_renderer();
    } catch (err) {
        setStatus(`WebGPU unavailable: ${err}`);
        console.error('init_renderer failed:', err);
        return;
    }

    setStatus('Choose a scene below, or drop a GLTF / GLB file');

    tick();
}

main().catch(err => {
    setStatus(`Unexpected error: ${err}`);
    console.error('main() failed:', err);
});
