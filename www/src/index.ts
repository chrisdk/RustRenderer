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

// ─────────────────────────────────────────────────────────────────────────────
// Camera state
// ─────────────────────────────────────────────────────────────────────────────

const MAX_PITCH = Math.PI / 2 - 0.001;  // matches Rust constant
const VFOV      = Math.PI / 3;           // 60° vertical FOV
const MOVE_SPEED = 0.08;                 // world units per frame
const LOOK_SPEED = 0.002;               // radians per pixel of mouse movement

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
});

/**
 * Process held movement keys for one frame. Returns true if the camera moved.
 */
function processMovement(): boolean {
    if (keysDown.size === 0) return false;

    const { fwd, right, up } = cameraBasis();
    let moved = false;

    if (keysDown.has('KeyW'))      { addScaled(camera.position, fwd,    MOVE_SPEED); moved = true; }
    if (keysDown.has('KeyS'))      { addScaled(camera.position, fwd,   -MOVE_SPEED); moved = true; }
    if (keysDown.has('KeyD'))      { addScaled(camera.position, right,  MOVE_SPEED); moved = true; }
    if (keysDown.has('KeyA'))      { addScaled(camera.position, right, -MOVE_SPEED); moved = true; }
    if (keysDown.has('Space'))     { addScaled(camera.position, up,     MOVE_SPEED); moved = true; }
    if (keysDown.has('ShiftLeft')) { addScaled(camera.position, up,    -MOVE_SPEED); moved = true; }

    return moved;
}

// ─────────────────────────────────────────────────────────────────────────────
// Canvas sizing
// ─────────────────────────────────────────────────────────────────────────────

function resizeCanvas(): void {
    // Match the canvas's backing-store resolution to its CSS display size
    // (which is 100vw × 100vh set in CSS). devicePixelRatio isn't used here
    // because the path tracer resolution is already bounded by GPU perf.
    const w = window.innerWidth;
    const h = window.innerHeight;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width  = w;
        canvas.height = h;
        cameraDirty   = true;   // output buffer dimensions changed
    }
}

window.addEventListener('resize', () => { resizeCanvas(); });
resizeCanvas();

// ─────────────────────────────────────────────────────────────────────────────
// Render loop
// ─────────────────────────────────────────────────────────────────────────────

let sceneLoaded  = false;
let cameraDirty  = false;
let rendering    = false;   // true while an async get_pixels call is in flight

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
    render(w, h);

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
 */
function tick(): void {
    if (processMovement()) cameraDirty = true;

    if (cameraDirty && sceneLoaded && !rendering) {
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
        // Apply the scene's recommended camera position, or leave the current
        // camera where it is if no preset was supplied (e.g. drag-and-drop).
        if (cameraPreset) {
            camera.position = [...cameraPreset.position];
            camera.yaw      = cameraPreset.yaw;
            camera.pitch    = cameraPreset.pitch;
        }
        sceneLoaded  = true;
        cameraDirty  = true;
        setStatus('Scene loaded — click to look around');
        fadeHints();
    } catch (err) {
        setStatus(`Error: ${err}`);
    }
}

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

            const glb = scene.build();
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
    await init();

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
        return;
    }

    setStatus('Choose a scene below, or drop a GLTF / GLB file');

    // 4. Start the animation loop.
    tick();
}

main();
