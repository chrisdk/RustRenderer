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
    unload_environment,
    set_ibl_enabled,
    set_env_background,
    set_max_bounces,
    set_sun_azimuth,
    set_sun_elevation,
    set_sun_intensity,
    set_ibl_scale,
    set_exposure,
    update_camera,
    render,
    get_pixels,
    raster_frame,
    raster_get_pixels,
    get_scene_bounds,
} from '../pkg/render.js';
import { BUILTIN_SCENES } from './scenes.js';
import { Turntable } from './turntable.js';
import { RenderController } from './render-controller.js';

// ─────────────────────────────────────────────────────────────────────────────
// DOM references
// ─────────────────────────────────────────────────────────────────────────────

const canvas        = document.getElementById('canvas')         as HTMLCanvasElement;
const ctx           = canvas.getContext('2d')!;
const statusEl      = document.getElementById('status')!;
const fileInput     = document.getElementById('file-input')     as HTMLInputElement;
const envInput      = document.getElementById('env-file-input') as HTMLInputElement;
const iblToggle     = document.getElementById('ibl-toggle')     as HTMLInputElement;
const envBgToggle   = document.getElementById('env-bg-toggle')  as HTMLInputElement;
const hintsEl       = document.getElementById('hints')!;
const dropOverlay   = document.getElementById('drop-overlay')!;
const renderBtn     = document.getElementById('render-btn')     as HTMLButtonElement;
const saveBtn       = document.getElementById('save-btn')       as HTMLButtonElement;
const resetCamBtn   = document.getElementById('reset-cam-btn')  as HTMLButtonElement;
const kbdHelpBtn    = document.getElementById('kbd-help-btn')   as HTMLButtonElement;
const kbdOverlay    = document.getElementById('kbd-overlay')!;
const sceneSelect   = document.getElementById('scene-select')   as HTMLSelectElement;
const sceneAttrib   = document.getElementById('scene-attribution')!;
const envSelect     = document.getElementById('env-select')     as HTMLSelectElement;
const sampleCount   = document.getElementById('sample-count')   as HTMLSelectElement;
const fovSlider     = document.getElementById('fov-slider')     as HTMLInputElement;
const fovValue      = document.getElementById('fov-value')!;
const bouncesSlider  = document.getElementById('bounces-slider')   as HTMLInputElement;
const bouncesValue   = document.getElementById('bounces-value')!;
const exposureSlider = document.getElementById('exposure-slider')  as HTMLInputElement;
const exposureValue  = document.getElementById('exposure-value')!;
const sunAzSlider    = document.getElementById('sun-az-slider')    as HTMLInputElement;
const sunAzValue     = document.getElementById('sun-az-value')!;
const sunElSlider    = document.getElementById('sun-el-slider')    as HTMLInputElement;
const sunElValue     = document.getElementById('sun-el-value')!;
const sunIntSlider   = document.getElementById('sun-int-slider')   as HTMLInputElement;
const sunIntValue    = document.getElementById('sun-int-value')!;
const iblScaleSlider = document.getElementById('ibl-scale-slider') as HTMLInputElement;
const iblScaleValue  = document.getElementById('ibl-scale-value')!;

// ─────────────────────────────────────────────────────────────────────────────
// Turntable camera
// ─────────────────────────────────────────────────────────────────────────────

const turntable = new Turntable();

let   vfov       = Math.PI / 3;   // vertical field of view in radians; mutable — driven by the FOV slider
const DRAG_SPEED = 0.005;         // radians of rotation per pixel dragged
const ZOOM_SPEED = 0.002;         // exponential zoom rate per scroll pixel

// The most recently loaded scene bounds, kept so the camera-reset button and
// the F shortcut can re-run autoFrame without reloading the scene.
let lastBounds: Float32Array | null = null;

/**
 * Uploads the current turntable state to the GPU and marks the camera dirty
 * so the render loop draws a fresh raster frame.
 */
function applyTurntable(): void {
    const { px, py, pz, yaw, pitch } = turntable.toCameraParams();
    const w = canvas.width  || 1;
    const h = canvas.height || 1;
    update_camera(px, py, pz, yaw, pitch, vfov, w / h);
    ctrl.cameraDirty = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Render controller
// ─────────────────────────────────────────────────────────────────────────────

const ctrl = new RenderController({
    load_scene,
    get_scene_bounds,
    render,
    get_pixels,
    raster_frame,
    raster_get_pixels,
    putImageData: (pixels, w, h) =>
        ctx.putImageData(new ImageData(new Uint8ClampedArray(pixels), w, h), 0, 0),
    getCanvasSize: () => ({ w: canvas.width || 1, h: canvas.height || 1 }),
    setStatus,
    onSceneLoaded: (bounds) => {
        lastBounds = bounds;
        turntable.autoFrame(bounds, vfov);
        applyTurntable();
        renderBtn.disabled   = false;
        saveBtn.disabled     = false;
        resetCamBtn.disabled = false;
        fadeHints();
    },
    lockCamera: (w, h) => {
        // Lock the viewpoint for the full HQ render. Does NOT set cameraDirty
        // so a raster frame isn't triggered on top of the path-traced output.
        const { px, py, pz, yaw, pitch } = turntable.toCameraParams();
        update_camera(px, py, pz, yaw, pitch, vfov, w / h);
    },
    onHqStart: () => {
        renderBtn.textContent = 'Cancel';
        renderBtn.classList.add('cancel');
    },
    onHqEnd: (_cancelled) => {
        renderBtn.textContent = 'Render';
        renderBtn.classList.remove('cancel');
    },
});

// ─────────────────────────────────────────────────────────────────────────────
// Drag input
// ─────────────────────────────────────────────────────────────────────────────

let dragging     = false;
let lastX        = 0;
let lastY        = 0;

canvas.addEventListener('pointerdown', e => {
    if (!ctrl.sceneLoaded) return;
    canvas.setPointerCapture(e.pointerId);
    dragging = true;
    lastX    = e.clientX;
    lastY    = e.clientY;
    canvas.classList.add('dragging');
    ctrl.cancelHighQualityRender();
});

canvas.addEventListener('pointermove', e => {
    if (!dragging) return;

    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    turntable.pan(-dx * DRAG_SPEED, dy * DRAG_SPEED);
    applyTurntable();
});

canvas.addEventListener('pointerup',     () => endDrag());
canvas.addEventListener('pointercancel', () => endDrag());

function endDrag(): void {
    dragging = false;
    canvas.classList.remove('dragging');
}

// ─────────────────────────────────────────────────────────────────────────────
// Scroll / pinch zoom
// ─────────────────────────────────────────────────────────────────────────────

canvas.addEventListener('wheel', e => {
    if (!ctrl.sceneLoaded) return;

    // Prevent the browser from intercepting the scroll (page scroll or the
    // native two-finger pinch-zoom overlay on macOS). Must be non-passive.
    e.preventDefault();

    // Normalise deltaY to "pixel-equivalent" units.  Browsers report three
    // modes: pixels (0, used by smooth touchpads), lines (1, classic mouse
    // wheels in Firefox), and pages (2, rare).  Multiply up so ZOOM_SPEED
    // only needs one tuning value.
    let delta = e.deltaY;
    if (e.deltaMode === 1 /* DOM_DELTA_LINE */) delta *= 16;
    if (e.deltaMode === 2 /* DOM_DELTA_PAGE */) delta *= 400;

    // Exponential zoom: equal up/down scrolls cancel exactly; feel is
    // proportional at any distance.
    //   deltaY < 0  →  scroll up / pinch open  →  zoom in
    //   deltaY > 0  →  scroll down / pinch in  →  zoom out
    turntable.zoom(Math.exp(delta * ZOOM_SPEED));

    ctrl.cancelHighQualityRender();
    applyTurntable();
}, { passive: false }); // passive: false is required to call preventDefault()

// ─────────────────────────────────────────────────────────────────────────────
// Canvas sizing
// ─────────────────────────────────────────────────────────────────────────────

function resizeCanvas(): void {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width  = w;
        canvas.height = h;
        if (ctrl.sceneLoaded) {
            // Update the camera projection with the new aspect ratio, then
            // mark the camera dirty so the raster loop redraws immediately.
            // Without this, the first post-resize frame uses the stale aspect
            // ratio and the scene appears stretched or squished.
            applyTurntable();
        } else {
            ctrl.cameraDirty = true;
        }
    }
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// ─────────────────────────────────────────────────────────────────────────────
// Main animation loop
// ─────────────────────────────────────────────────────────────────────────────

// ── Display strategy ──────────────────────────────────────────────────────────
//
// The rasterizer is the primary display: it renders whenever the camera changes
// (drag, resize, scene load) and its output stays on screen until something
// replaces it. The path tracer is only used for the explicit "Render" button.
//
// This means the canvas always shows a sharp, textured, correctly-lit image
// even when idle — no progressive noise build-up before the user renders.

/**
 * Main animation loop.
 *
 * Renders a raster frame whenever the camera changes (drag, resize, scene
 * load). The rasterizer output stays on screen until the HQ path tracer
 * explicitly replaces it — so the canvas is always showing something useful.
 */
function tick(): void {
    if (ctrl.shouldRenderDragFrame()) {
        ctrl.cameraDirty = false;
        ctrl.renderDragFrame();
    }

    requestAnimationFrame(tick);
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene / environment loading
// ─────────────────────────────────────────────────────────────────────────────

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

// ── Render button ─────────────────────────────────────────────────────────────

renderBtn.addEventListener('click', () => {
    if (ctrl.hqRendering) {
        ctrl.cancelHighQualityRender();
    } else {
        ctrl.startHighQualityRender();
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// Tab switching
// ─────────────────────────────────────────────────────────────────────────────

document.querySelectorAll<HTMLButtonElement>('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`tab-${btn.dataset.tab}`)?.classList.add('active');
    });
});

// ─────────────────────────────────────────────────────────────────────────────
// Scene dropdown
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Populates the scene `<select>` from BUILTIN_SCENES and wires the change
 * handler to download and load the chosen scene.
 *
 * Attribution text for licensed assets (CC BY 4.0 etc.) is shown below the
 * dropdown when a non-CC0 scene is selected, and cleared for CC0 assets.
 */
function buildSceneDropdown(): void {
    for (const scene of BUILTIN_SCENES) {
        const opt = document.createElement('option');
        opt.value       = scene.name;
        opt.textContent = scene.label;
        sceneSelect.appendChild(opt);
    }

    sceneSelect.addEventListener('change', async () => {
        const scene = BUILTIN_SCENES.find(s => s.name === sceneSelect.value);
        if (!scene) return;

        // Show attribution for non-CC0 assets; clear it for public-domain ones.
        sceneAttrib.textContent = scene.attribution ?? '';

        setStatus('Downloading scene…');
        const glb = await Promise.resolve(scene.build());
        await ctrl.loadSceneBytes(new Uint8Array(glb), scene.name);
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Environment dropdown
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Wires the environment `<select>` dropdown.
 *
 * Three kinds of options:
 *   - `""` (Procedural sky) — calls unload_environment() and reverts to the
 *     built-in sky gradient.
 *   - Named .hdr file — fetches it from the `assets/` directory.
 *   - `"__file__"` (Load from file…) — programmatically clicks the hidden
 *     file input; the select value reverts so the picker reads "Load from
 *     file…" only while the native dialog is open.
 *
 * After a custom file is loaded, a "Custom: <filename>" option is
 * inserted/replaced so the dropdown reflects the active environment.
 */
function buildEnvDropdown(): void {
    // Track the last successfully applied env value so we can revert the
    // select when "Load from file…" is chosen but the user cancels the picker.
    let lastEnvValue = envSelect.value;  // starts at "" (Procedural sky)

    envSelect.addEventListener('change', async () => {
        const val = envSelect.value;

        if (val === '__file__') {
            // Revert to the previous choice; the real change happens when the
            // file picker fires its 'change' event below.
            envSelect.value = lastEnvValue;
            envInput.click();
            return;
        }

        lastEnvValue = val;

        if (val === '') {
            // Revert to procedural sky — clear any loaded HDR.
            unload_environment();
            ctrl.cancelHighQualityRender();
            setStatus('Using procedural sky');
        } else {
            // Fetch a named built-in HDR from the assets directory.
            setStatus('Loading environment…');
            try {
                const r = await fetch(`assets/${val}`);
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                const buf = await r.arrayBuffer();
                await loadEnvironmentBytes(new Uint8Array(buf));
            } catch (err) {
                setStatus(`Environment error: ${err}`);
            }
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// File pickers
// ─────────────────────────────────────────────────────────────────────────────

fileInput.addEventListener('change', async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    const buffer = await file.arrayBuffer();
    await ctrl.loadSceneBytes(new Uint8Array(buffer), `${file.name}:${file.size}`);
});

envInput.addEventListener('change', async () => {
    const file = envInput.files?.[0];
    if (!file) return;
    const buffer = await file.arrayBuffer();
    await loadEnvironmentBytes(new Uint8Array(buffer));

    // Insert or replace the "Custom: <name>" option so the dropdown reflects
    // the file that was just loaded.
    const CUSTOM_VALUE = '__custom__';
    let customOpt = envSelect.querySelector<HTMLOptionElement>(`option[value="${CUSTOM_VALUE}"]`);
    if (!customOpt) {
        customOpt = document.createElement('option');
        customOpt.value = CUSTOM_VALUE;
        // Insert before the separator (the disabled "──────────" option).
        const separator = envSelect.querySelector<HTMLOptionElement>('option[disabled]');
        envSelect.insertBefore(customOpt, separator ?? null);
    }
    customOpt.textContent = `Custom: ${file.name}`;
    envSelect.value = CUSTOM_VALUE;
});

// ── Drag-and-drop ─────────────────────────────────────────────────────────────

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
        await ctrl.loadSceneBytes(new Uint8Array(buffer), `${file.name}:${file.size}`);
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// IBL toggle
// ─────────────────────────────────────────────────────────────────────────────

iblToggle.addEventListener('change', () => {
    set_ibl_enabled(iblToggle.checked);
    // Cancelling the HQ render causes it to restart from scratch on the next
    // Render click — necessary because the IBL toggle changes the lighting and
    // would otherwise blend old (no IBL) and new (IBL) samples together.
    ctrl.cancelHighQualityRender();
});

// ─────────────────────────────────────────────────────────────────────────────
// Environment background toggle
// ─────────────────────────────────────────────────────────────────────────────

envBgToggle.addEventListener('change', () => {
    set_env_background(envBgToggle.checked);
    // Same reason as the IBL toggle: changing the background invalidates any
    // accumulated samples — old ones used the env map, new ones would use dark
    // grey, so blending them gives garbage.
    ctrl.cancelHighQualityRender();
});

// ─────────────────────────────────────────────────────────────────────────────
// Sample-count selector
// ─────────────────────────────────────────────────────────────────────────────

sampleCount.addEventListener('change', () => {
    ctrl.hqSamples = parseInt(sampleCount.value, 10);
});

// ─────────────────────────────────────────────────────────────────────────────
// FOV slider
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Live-updates the vertical field of view as the slider moves.
 *
 * The FOV only affects the camera projection — no GPU scene data changes, so
 * no re-upload is needed. `applyTurntable()` calls `update_camera()` with the
 * new angle; the rasterizer redraws immediately. Any in-progress HQ render is
 * cancelled because old samples were traced with the old FOV and would blend
 * incorrectly with new ones.
 *
 * Note: the slider changes the view angle but does NOT auto-reframe the camera
 * — the turntable position stays fixed so the change looks like a lens swap
 * rather than a sudden jump. The next camera reset (F or "Reset camera") will
 * auto-frame using the new FOV.
 */
fovSlider.addEventListener('input', () => {
    const degrees = parseInt(fovSlider.value, 10);
    vfov = degrees * Math.PI / 180;
    fovValue.textContent = `${degrees}°`;
    ctrl.cancelHighQualityRender();
    applyTurntable();
});

// ─────────────────────────────────────────────────────────────────────────────
// Renderer controls — bounce count, exposure, sun, IBL
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Max-bounces slider. Higher = more accurate glass/indirect, slower render.
 * Starts a new render because old samples (traced at a different depth) would
 * blend incorrectly with new ones.
 */
bouncesSlider.addEventListener('input', () => {
    const n = parseInt(bouncesSlider.value, 10);
    bouncesValue.textContent = `${n}`;
    set_max_bounces(n);
    ctrl.cancelHighQualityRender();
});

/**
 * Exposure slider. Applied as `linear × 2^EV` before ACES tone-mapping.
 *
 * Exposure is a display-time transform over the already-accumulated average,
 * so it does NOT need to restart the render — the same samples look brighter
 * or dimmer on the next `get_pixels` call. This makes it the only slider that
 * works live during a running HQ render.
 *
 * The range element uses integer steps (−30…+30) to avoid float-step browser
 * quirks; we divide by 10 to get −3.0…+3.0 EV.
 */
exposureSlider.addEventListener('input', () => {
    const ev = parseInt(exposureSlider.value, 10) / 10;
    exposureValue.textContent = `${ev >= 0 ? '+' : ''}${ev.toFixed(1)} EV`;
    set_exposure(ev);
    // Intentionally no cancelHighQualityRender() — exposure updates live.
});

/**
 * Sun azimuth slider — horizontal compass direction (0–360°).
 * Restarts any running HQ render because the lighting changes.
 */
sunAzSlider.addEventListener('input', () => {
    const deg = parseInt(sunAzSlider.value, 10);
    sunAzValue.textContent = `${deg}°`;
    set_sun_azimuth(deg);
    ctrl.cancelHighQualityRender();
});

/**
 * Sun elevation slider — angle above the horizon (0 = horizon, 90 = zenith).
 */
sunElSlider.addEventListener('input', () => {
    const deg = parseInt(sunElSlider.value, 10);
    sunElValue.textContent = `${deg}°`;
    set_sun_elevation(deg);
    ctrl.cancelHighQualityRender();
});

/**
 * Sun intensity slider. Integer steps ×10 → 0.0–5.0× multiplier.
 */
sunIntSlider.addEventListener('input', () => {
    const scale = parseInt(sunIntSlider.value, 10) / 10;
    sunIntValue.textContent = `${scale.toFixed(1)}×`;
    set_sun_intensity(scale);
    ctrl.cancelHighQualityRender();
});

/**
 * IBL brightness slider. Scales env-map and sky contributions uniformly.
 * Integer steps ×10 → 0.0–4.0× multiplier.
 */
iblScaleSlider.addEventListener('input', () => {
    const scale = parseInt(iblScaleSlider.value, 10) / 10;
    iblScaleValue.textContent = `${scale.toFixed(1)}×`;
    set_ibl_scale(scale);
    ctrl.cancelHighQualityRender();
});

// ─────────────────────────────────────────────────────────────────────────────
// Save PNG
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Exports the current canvas contents as a PNG download.
 *
 * `canvas.toBlob` is asynchronous (the browser encodes in the background) so
 * we create a temporary object URL and click a synthetic <a> to trigger the
 * browser's native Save dialog. The URL is revoked immediately after the click
 * so the Blob is eligible for GC.
 */
function savePng(): void {
    canvas.toBlob(blob => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a   = document.createElement('a');
        a.href     = url;
        a.download = 'render.png';
        a.click();
        URL.revokeObjectURL(url);
    }, 'image/png');
}

saveBtn.addEventListener('click', savePng);

// ─────────────────────────────────────────────────────────────────────────────
// Camera reset
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Returns the camera to the default framing for the current scene.
 *
 * Uses the stored scene bounds from the last `onSceneLoaded` callback so we
 * don't need to round-trip to the GPU just to get the AABB again.
 */
function resetCamera(): void {
    if (!lastBounds) return;
    ctrl.cancelHighQualityRender();
    turntable.autoFrame(lastBounds, vfov);
    applyTurntable();
}

resetCamBtn.addEventListener('click', resetCamera);

// ─────────────────────────────────────────────────────────────────────────────
// Keyboard shortcuts
// ─────────────────────────────────────────────────────────────────────────────

// Zoom factor per keypress — 15 % change, matching a comfortable scroll notch.
const KB_ZOOM_IN  = 1 / 1.15;   // smaller radius = closer
const KB_ZOOM_OUT = 1.15;        // larger radius  = farther

window.addEventListener('keydown', e => {
    // Don't intercept shortcuts when the user is typing in a form element.
    const tag = (e.target as HTMLElement).tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

    switch (e.key) {
        case 'r':
        case 'R':
            if (!ctrl.sceneLoaded) break;
            if (ctrl.hqRendering) {
                ctrl.cancelHighQualityRender();
            } else {
                ctrl.startHighQualityRender();
            }
            break;

        case 'Escape':
            ctrl.cancelHighQualityRender();
            break;

        case 'f':
        case 'F':
            resetCamera();
            break;

        // + / = both map to "zoom in" — = is the unshifted key on most layouts.
        case '+':
        case '=':
            if (!ctrl.sceneLoaded) break;
            turntable.zoom(KB_ZOOM_IN);
            ctrl.cancelHighQualityRender();
            applyTurntable();
            break;

        case '-':
        case '_':
            if (!ctrl.sceneLoaded) break;
            turntable.zoom(KB_ZOOM_OUT);
            ctrl.cancelHighQualityRender();
            applyTurntable();
            break;

        case '?':
            kbdOverlay.hidden = !kbdOverlay.hidden;
            break;
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// Keyboard help overlay
// ─────────────────────────────────────────────────────────────────────────────

kbdHelpBtn.addEventListener('click', () => {
    kbdOverlay.hidden = !kbdOverlay.hidden;
});

// Clicking anywhere on the canvas dismisses the overlay so it doesn't block
// the view when the user starts interacting.
canvas.addEventListener('pointerdown', () => {
    kbdOverlay.hidden = true;
}, { capture: true });

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

    buildSceneDropdown();
    buildEnvDropdown();

    setStatus('Requesting GPU…');
    try {
        await init_renderer();
    } catch (err) {
        setStatus(`WebGPU unavailable: ${err}`);
        console.error('init_renderer failed:', err);
        return;
    }

    // Pre-load the default scene (Damaged Helmet) in the background.
    // Programmatic .value assignment doesn't fire 'change', so we trigger
    // the load explicitly and mirror the same pattern as the env map below.
    const defaultScene = BUILTIN_SCENES[0];
    sceneSelect.value = defaultScene.name;
    sceneAttrib.textContent = defaultScene.attribution ?? '';
    setStatus('Downloading scene…');
    Promise.resolve(defaultScene.build())
        .then(glb => ctrl.loadSceneBytes(new Uint8Array(glb), defaultScene.name))
        .catch(err => {
            console.warn('Default scene not loaded:', err);
            // Revert to placeholder so the user knows they need to choose.
            sceneSelect.value = '';
            sceneAttrib.textContent = '';
            setStatus('Choose a scene, or drop a GLTF / GLB file');
        });

    // Kick off the default env map fetch in the background — the renderer
    // is perfectly usable without it; it just falls back to procedural sky.
    // We don't await this so the rest of the UI comes up immediately.
    // Reflect the pending load in the dropdown so the user sees what's loading.
    envSelect.value = 'golden_gate_hills_2k.hdr';
    fetch('assets/golden_gate_hills_2k.hdr')
        .then(r => {
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            return r.arrayBuffer();
        })
        .then(buf => loadEnvironmentBytes(new Uint8Array(buf)))
        .catch(err => {
            console.warn('Default env map not loaded:', err);
            // Revert the dropdown to "Procedural sky" since nothing loaded.
            envSelect.value = '';
        });

    tick();
}

main().catch(err => {
    setStatus(`Unexpected error: ${err}`);
    console.error('main() failed:', err);
});
