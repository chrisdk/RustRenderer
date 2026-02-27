/**
 * Path tracer frontend.
 *
 * Responsibilities:
 *   1. Load the WASM module and initialise the GPU renderer.
 *   2. Accept a GLTF/GLB scene via drag-and-drop or file picker.
 *   3. Run an FPS camera: pointer-lock mouse-look + WASD/Space/Shift movement.
 *   4. Each frame: update_camera → render → get_pixels → putImageData on canvas.
 */

import init, {
    init_renderer,
    load_scene,
    update_camera,
    render,
    get_pixels,
    get_scene_bounds,
} from '../pkg/render.js';
import { BUILTIN_SCENES } from './scenes.js';

// ─────────────────────────────────────────────────────────────────────────────
// DOM references
// ─────────────────────────────────────────────────────────────────────────────

const canvas       = document.getElementById('canvas')      as HTMLCanvasElement;
const ctx          = canvas.getContext('2d')!;
const statusEl     = document.getElementById('status')!;
const fileInput    = document.getElementById('file-input')  as HTMLInputElement;
const hintsEl      = document.getElementById('hints')!;
const dropOverlay  = document.getElementById('drop-overlay')!;
const scenePicker  = document.getElementById('scene-picker')!;
const renderBtn    = document.getElementById('render-btn')  as HTMLButtonElement;
const speedDownBtn = document.getElementById('speed-down')  as HTMLButtonElement;
const speedUpBtn   = document.getElementById('speed-up')    as HTMLButtonElement;
const speedDisplay = document.getElementById('speed-display')!;

// ─────────────────────────────────────────────────────────────────────────────
// Camera state
// ─────────────────────────────────────────────────────────────────────────────

const MAX_PITCH  = Math.PI / 2 - 0.001;  // matches Rust constant
const VFOV       = Math.PI / 3;           // 60° vertical FOV
const LOOK_SPEED = 0.002;                // radians per pixel of mouse movement

// ── Move speed ────────────────────────────────────────────────────────────────
// baseSpeed is computed from the scene's bounding-box diagonal so that
// navigating any scene feels similarly fluid regardless of real-world scale.
// The user can then nudge it up or down with the panel widget.
//
// Multiplier steps span six doublings so the user can cover the full range
// from "creeping around a tiny detail" to "flying across a huge environment"
// without needing a text field.

const SPEED_STEPS = [0.25, 0.5, 1, 2, 4, 8];
let speedStepIdx  = 2;          // default: 1× (index into SPEED_STEPS)
let baseSpeed     = 0.08;       // auto-set per scene; fallback for pre-load
let moveSpeed     = baseSpeed;  // actual value used each frame

const camera = {
    position: [0, 1, 5] as [number, number, number],
    yaw:   0,
    pitch: 0,
};

/**
 * Compute the camera's orthonormal basis vectors from yaw and pitch.
 * Mirrors Camera::basis() in camera.rs exactly.
 */
function cameraBasis(): { fwd: Vec3; right: Vec3; up: Vec3 } {
    const sy = Math.sin(camera.yaw),  cy = Math.cos(camera.yaw);
    const sp = Math.sin(camera.pitch), cp = Math.cos(camera.pitch);

    const fwd:   Vec3 = [sy * cp, sp, -cy * cp];
    const right: Vec3 = [cy, 0, sy];
    // up = cross(right, fwd) — always horizontal right since right.y = 0.
    const up: Vec3 = [
        right[1] * fwd[2] - right[2] * fwd[1],
        right[2] * fwd[0] - right[0] * fwd[2],
        right[0] * fwd[1] - right[1] * fwd[0],
    ];
    return { fwd, right, up };
}

type Vec3 = [number, number, number];
function addScaled(out: Vec3, v: Vec3, s: number): void {
    out[0] += v[0] * s;
    out[1] += v[1] * s;
    out[2] += v[2] * s;
}

/**
 * Position the camera so the loaded scene fills roughly 80% of the viewport
 * in whichever axis (horizontal or vertical) it's largest in screen space.
 *
 * The camera is placed along +Z looking toward the scene centre, at a distance
 * computed from the bounding box half-extents and the current vFOV/aspect:
 *
 *   dist = max(
 *     hy / (0.8 * tan(vfov/2)),          // fill 80% vertically
 *     hx / (0.8 * tan(vfov/2) * aspect), // fill 80% horizontally
 *     hz * 1.1,                          // stay outside the model's depth
 *   )
 *
 * Falls back silently if no scene bounds are available yet.
 */
function applyAutoCamera(): void {
    const b = get_scene_bounds();
    if (b.length < 6) return;

    const cx = (b[0] + b[3]) / 2;
    const cy = (b[1] + b[4]) / 2;
    const cz = (b[2] + b[5]) / 2;
    const hx = (b[3] - b[0]) / 2;
    const hy = (b[4] - b[1]) / 2;
    const hz = (b[5] - b[2]) / 2;

    const aspect = canvas.width > 0 && canvas.height > 0
        ? canvas.width / canvas.height
        : 16 / 9;
    const halfTanVFov = Math.tan(VFOV / 2);

    const dByHeight = hy / (0.8 * halfTanVFov);
    const dByWidth  = hx / (0.8 * halfTanVFov * aspect);
    const dist = Math.max(dByHeight, dByWidth, hz * 1.1, 0.1);

    camera.position = [cx, cy, cz + dist];
    camera.yaw      = 0;
    camera.pitch    = 0;
}

/**
 * Set the base movement speed from the loaded scene's bounding box.
 *
 * We target "traverse the full scene diagonal in ~100 frames" at the 1×
 * multiplier setting (about 1.7 s at 60 fps). That feels responsive for
 * close-up inspection without being a blur for larger environments.
 *
 * The user's current multiplier step is preserved across scene loads so that
 * someone who cranked it up for a big scene doesn't have to re-adjust when
 * swapping to a smaller one — only the baseline changes.
 */
function applyAutoSpeed(): void {
    const b = get_scene_bounds();
    if (b.length < 6) return;

    const dx = b[3] - b[0];
    const dy = b[4] - b[1];
    const dz = b[5] - b[2];
    const diagonal = Math.sqrt(dx * dx + dy * dy + dz * dz);

    // 1% of the scene diagonal per frame = full diagonal in 100 frames.
    baseSpeed = Math.max(diagonal * 0.01, 1e-4);
    moveSpeed = baseSpeed * SPEED_STEPS[speedStepIdx];
    updateSpeedDisplay();
}

/** Refresh the speed readout and button states in the panel. */
function updateSpeedDisplay(): void {
    speedDisplay.textContent = `${SPEED_STEPS[speedStepIdx]}×`;
    speedDownBtn.disabled = speedStepIdx === 0;
    speedUpBtn.disabled   = speedStepIdx === SPEED_STEPS.length - 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// Input handling
// ─────────────────────────────────────────────────────────────────────────────

const keysDown = new Set<string>();

window.addEventListener('keydown', e => {
    keysDown.add(e.code);
    // Swallow spacebar so the page doesn't scroll.
    if (e.code === 'Space') e.preventDefault();
});
window.addEventListener('keyup',   e => keysDown.delete(e.code));

// Click the canvas to capture the mouse for look-around.
canvas.addEventListener('click', () => {
    if (sceneLoaded) canvas.requestPointerLock();
});

document.addEventListener('mousemove', e => {
    if (document.pointerLockElement !== canvas) return;
    camera.yaw   += e.movementX * LOOK_SPEED;
    camera.pitch  = Math.max(-MAX_PITCH,
                    Math.min( MAX_PITCH,
                    camera.pitch - e.movementY * LOOK_SPEED));
    cameraDirty = true;
    if (hqRendering) hqCancelled = true;  // camera moved — abort the HQ render
});

/**
 * Process held movement keys for one frame. Returns true if the camera moved.
 */
function processMovement(): boolean {
    if (keysDown.size === 0) return false;

    const { fwd, right, up } = cameraBasis();
    let moved = false;

    if (keysDown.has('KeyW'))      { addScaled(camera.position, fwd,    moveSpeed); moved = true; }
    if (keysDown.has('KeyS'))      { addScaled(camera.position, fwd,   -moveSpeed); moved = true; }
    if (keysDown.has('KeyD'))      { addScaled(camera.position, right,  moveSpeed); moved = true; }
    if (keysDown.has('KeyA'))      { addScaled(camera.position, right, -moveSpeed); moved = true; }
    if (keysDown.has('Space'))     { addScaled(camera.position, up,     moveSpeed); moved = true; }
    if (keysDown.has('ShiftLeft')) { addScaled(camera.position, up,    -moveSpeed); moved = true; }

    return moved;
}

// ─────────────────────────────────────────────────────────────────────────────
// Render loop state  (declared before resizeCanvas so the TDZ doesn't bite)
// ─────────────────────────────────────────────────────────────────────────────

let sceneLoaded  = false;
let cameraDirty  = false;
let rendering    = false;   // true while an async get_pixels call is in flight

// High-quality progressive render state.
let hqRendering  = false;   // true while the multi-sample render loop is running
let hqCancelled  = false;   // set to true to stop the loop after the current sample

const HQ_SAMPLES = 256;     // samples per pixel for a full quality render

// ─────────────────────────────────────────────────────────────────────────────
// Canvas sizing
// ─────────────────────────────────────────────────────────────────────────────

function resizeCanvas(): void {
    // Read the canvas's CSS-rendered size. The canvas element fills its
    // flex container (#canvas-wrap), which takes up the window minus the
    // control panel, so clientWidth/clientHeight give the right dimensions
    // without any hardcoded panel-width arithmetic.
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width  = w;
        canvas.height = h;
        cameraDirty   = true;   // output buffer dimensions changed
    }
}

window.addEventListener('resize', () => { resizeCanvas(); });
resizeCanvas();

/**
 * Dispatch a render, read pixels back from the GPU, and blit to the canvas.
 * Fires-and-forgets: the caller does not await this.
 */
async function renderFrame(): Promise<void> {
    if (rendering) return;
    rendering = true;

    const w = canvas.width;
    const h = canvas.height;

    // Push the new camera to the GPU.
    update_camera(
        camera.position[0], camera.position[1], camera.position[2],
        camera.yaw, camera.pitch,
        VFOV,
        w / h,
    );

    // Dispatch the compute shader (synchronous — just enqueues GPU work).
    // sample_index = 0 resets the accumulator, giving a clean single-sample
    // preview on every camera update.
    render(w, h, 0);

    // Read pixels back (async — waits for GPU completion).
    const pixels = await get_pixels(w, h);

    // ImageData expects Uint8ClampedArray backed by a plain ArrayBuffer.
    // Copying via the TypedArray constructor is the safest cross-browser path
    // (avoids SharedArrayBuffer compatibility issues that arise from wasm-bindgen
    // returning Uint8Array whose .buffer is typed as ArrayBufferLike).
    const clamped   = new Uint8ClampedArray(pixels);
    const imageData = new ImageData(clamped, w, h);
    ctx.putImageData(imageData, 0, 0);

    rendering = false;
}

/**
 * Main animation loop. Processes input and triggers a render whenever the
 * camera changes. Runs at the browser's display refresh rate.
 *
 * While a high-quality render is running, camera movement cancels it so the
 * user can immediately navigate to a better angle without waiting.
 */
function tick(): void {
    if (processMovement()) {
        cameraDirty = true;
        if (hqRendering) hqCancelled = true;
    }

    // Skip preview renders while the HQ render is accumulating samples.
    if (!hqRendering && cameraDirty && sceneLoaded && !rendering) {
        cameraDirty = false;
        renderFrame();  // intentionally not awaited
    }

    requestAnimationFrame(tick);
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene loading
// ─────────────────────────────────────────────────────────────────────────────

async function loadSceneBytes(
    bytes: Uint8Array,
    cameraPreset?: { position: [number, number, number]; yaw: number; pitch: number },
): Promise<void> {
    setStatus('Loading scene…');
    try {
        load_scene(bytes);
        // Use the explicit preset (e.g. Cornell Box needs an interior viewpoint),
        // otherwise auto-frame from the scene's bounding box.
        if (cameraPreset) {
            camera.position = [...cameraPreset.position];
            camera.yaw      = cameraPreset.yaw;
            camera.pitch    = cameraPreset.pitch;
        } else {
            applyAutoCamera();
        }
        applyAutoSpeed();
        sceneLoaded  = true;
        cameraDirty  = true;
        renderBtn.disabled = false;
        setStatus('Scene loaded — click to look around');
        fadeHints();
    } catch (err) {
        setStatus(`Error: ${err}`);
    }
}

// ── High-quality render ───────────────────────────────────────────────────────

/**
 * Progressively accumulate HQ_SAMPLES samples into the same image.
 *
 * Each iteration dispatches one sample and reads the result back to the
 * canvas, so the image visibly sharpens in real time. The loop runs until
 * it has accumulated HQ_SAMPLES samples, the user clicks Cancel, or the
 * camera moves.
 */
async function startHighQualityRender(): Promise<void> {
    if (!sceneLoaded || hqRendering) return;

    // Wait for any in-flight preview render to finish before taking over.
    while (rendering) {
        await new Promise<void>(r => setTimeout(r, 16));
    }

    hqRendering  = true;
    hqCancelled  = false;
    rendering    = true;   // block the preview render loop
    renderBtn.textContent = 'Cancel';
    renderBtn.classList.add('cancel');

    const w = canvas.width;
    const h = canvas.height;

    // Lock in the current camera for the full render.
    update_camera(
        camera.position[0], camera.position[1], camera.position[2],
        camera.yaw, camera.pitch, VFOV, w / h,
    );

    for (let i = 0; i < HQ_SAMPLES && !hqCancelled; i++) {
        render(w, h, i);
        const pixels  = await get_pixels(w, h);
        ctx.putImageData(new ImageData(new Uint8ClampedArray(pixels), w, h), 0, 0);
        setStatus(`Rendering… ${i + 1} / ${HQ_SAMPLES} spp`);
    }

    rendering   = false;
    hqRendering = false;
    renderBtn.textContent = 'Render';
    renderBtn.classList.remove('cancel');

    if (hqCancelled) {
        setStatus('Render cancelled — click to look around');
        cameraDirty = true;   // redraw preview at new camera position
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

speedDownBtn.addEventListener('click', () => {
    if (speedStepIdx > 0) {
        speedStepIdx--;
        moveSpeed = baseSpeed * SPEED_STEPS[speedStepIdx];
        updateSpeedDisplay();
    }
});

speedUpBtn.addEventListener('click', () => {
    if (speedStepIdx < SPEED_STEPS.length - 1) {
        speedStepIdx++;
        moveSpeed = baseSpeed * SPEED_STEPS[speedStepIdx];
        updateSpeedDisplay();
    }
});

// ── Built-in scene picker ─────────────────────────────────────────────────────

/**
 * Populate the scene picker with one button per built-in scene.
 * Called once after the WASM module is ready (buttons need the module loaded
 * before clicking them does anything useful).
 */
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

            // build() is synchronous for procedural scenes and returns a
            // Promise for remote scenes that need a network fetch.
            setStatus('Downloading scene…');
            const glb = await Promise.resolve(scene.build());
            await loadSceneBytes(new Uint8Array(glb), scene.camera);
        });

        scenePicker.appendChild(btn);
    }
}

// ── File picker ──────────────────────────────────────────────────────────────

// #load-btn is a <label for="file-input">, so the browser opens the file
// picker natively — no JS click forwarding needed.

fileInput.addEventListener('change', async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    const buffer = await file.arrayBuffer();
    await loadSceneBytes(new Uint8Array(buffer));
});

// ── Drag-and-drop ────────────────────────────────────────────────────────────

// Show the overlay while a file is being dragged over the window.
window.addEventListener('dragenter', e => {
    e.preventDefault();
    dropOverlay.classList.add('active');
});
window.addEventListener('dragover', e => {
    e.preventDefault();
});
window.addEventListener('dragleave', e => {
    // Only hide the overlay when leaving the window entirely.
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
    await loadSceneBytes(new Uint8Array(buffer));
});

// ─────────────────────────────────────────────────────────────────────────────
// UI helpers
// ─────────────────────────────────────────────────────────────────────────────

function setStatus(msg: string): void {
    statusEl.textContent = msg;
}

/** Fade the controls hint out after a short delay. */
function fadeHints(): void {
    setTimeout(() => hintsEl.classList.add('fade'), 4000);
}

// ─────────────────────────────────────────────────────────────────────────────
// Boot sequence
// ─────────────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
    // 1. Load the WASM binary.
    setStatus('Loading WASM…');
    try {
        await init();
    } catch (err) {
        setStatus(`WASM load failed: ${err}`);
        console.error('WASM init failed:', err);
        return;
    }

    // 2. Populate the scene picker. The buttons only generate GLB bytes and
    //    call load_scene() — they don't need a GPU device to be rendered into
    //    the DOM. Doing this before init_renderer() means the picker is always
    //    visible even if the GPU request fails.
    buildScenePicker();

    // 3. Acquire the WebGPU device (async — goes through the browser's GPU API).
    setStatus('Requesting GPU…');
    try {
        await init_renderer();
    } catch (err) {
        setStatus(`WebGPU unavailable: ${err}`);
        console.error('init_renderer failed:', err);
        return;
    }

    setStatus('Choose a scene below, or drop a GLTF / GLB file');

    // 4. Initialise the speed widget to its default state.
    updateSpeedDisplay();

    // 5. Start the animation loop.
    tick();
}

main().catch(err => {
    setStatus(`Unexpected error: ${err}`);
    console.error('main() failed:', err);
});
