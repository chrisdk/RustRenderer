/* tslint:disable */
/* eslint-disable */

/**
 * Reads the last rendered frame back from the GPU and returns it as a
 * `Uint8Array` of raw RGBA8 bytes, suitable for passing to `ImageData`
 * and drawing to a `<canvas>` element.
 *
 * Must be called after `render()`. Returns an empty array if no frame has
 * been rendered yet.
 */
export function get_pixels(width: number, height: number): Promise<Uint8Array>;

/**
 * Returns the world-space axis-aligned bounding box of the loaded scene as a
 * `Float32Array` of six values: `[min_x, min_y, min_z, max_x, max_y, max_z]`.
 *
 * Returns an empty array if no scene has been loaded yet. The frontend uses
 * this to auto-position the camera so the model fills the viewport sensibly.
 */
export function get_scene_bounds(): Float32Array;

/**
 * Returns a snapshot of the most recently loaded scene's statistics as a
 * `Uint32Array` of five values:
 * `[triangle_count, vertex_count, mesh_count, texture_count, file_size_kb]`.
 *
 * The file size is in whole kilobytes; divide by 1024 in JS to get MB.
 * Returns all-zeros if no scene has been loaded yet — the frontend shows
 * `—` for each field until a real scene arrives.
 */
export function get_scene_stats(): Uint32Array;

/**
 * Initialises the renderer: sets up logging, selects a GPU adapter, and opens
 * a WebGPU device.
 *
 * Must be called once before any other function. Returns a `Promise` that
 * resolves when the GPU device is ready, or rejects with an error string if
 * WebGPU is not supported or device creation fails.
 */
export function init_renderer(): Promise<void>;

/**
 * Loads an HDR environment map from raw `.hdr` (Radiance RGBE) bytes.
 *
 * The environment map is used by both the path tracer and the rasteriser
 * preview for background sky colour, ambient lighting, and reflections in
 * metallic surfaces. Call this after `init_renderer`. Can be called again
 * to swap environments without reloading the scene.
 */
export function load_environment(bytes: Uint8Array): void;

/**
 * Parses a GLTF/GLB byte array, builds the BVH, and uploads all scene data
 * to the GPU storage buffers.
 *
 * Must be called after `init_renderer`. Can be called again to swap scenes;
 * the old GPU buffers are dropped and replaced.
 */
export function load_scene(bytes: Uint8Array): void;

/**
 * Draws the scene using the fast rasterising preview pipeline.
 *
 * This runs at 60 fps during interactive dragging. Unlike the path tracer
 * (`render`), it produces a single fully-formed frame with no progressive
 * accumulation — call `raster_get_pixels` immediately after to retrieve it.
 *
 * All user-adjustable controls (sun azimuth/elevation, sun intensity, exposure,
 * IBL scale, IBL enabled/disabled, and any loaded HDR env map) are reflected in
 * the preview frame just as they are in the path-traced output.
 *
 * Requires both `init_renderer` and `load_scene` to have been called first.
 */
export function raster_frame(width: number, height: number): void;

/**
 * Reads the last rasterised frame back from the GPU and returns it as a
 * `Uint8Array` of raw RGBA8 bytes.
 *
 * Must be called immediately after `raster_frame()`.
 */
export function raster_get_pixels(width: number, height: number): Promise<Uint8Array>;

/**
 * Dispatches the path-tracing compute shader for one sample pass.
 *
 * `sample_index` identifies which sample this is in a progressive render:
 * - Pass `0` to start a new render (resets the accumulator).
 * - Pass `1, 2, 3, …` to accumulate additional samples into the same image.
 *
 * The output buffer holds the running average after each dispatch, so calling
 * `get_pixels` after any sample gives a valid (progressively improving) image.
 *
 * Requires both `init_renderer` and `load_scene` to have been called first.
 * Calling before the scene is loaded is a no-op with a console warning.
 */
export function render(width: number, height: number, sample_index: number, preview: boolean): void;

/**
 * Sets the thin-lens aperture radius in world units.
 *
 * `0.0` (the default) gives a pinhole camera where everything is in focus.
 * Larger values widen the circle of confusion for geometry that is not on
 * the focal plane, producing cinematic bokeh. Changes take effect on the
 * next `render()` call with `sample_index = 0` (a fresh accumulation).
 */
export function set_aperture(radius: number): void;

/**
 * Controls whether the loaded environment map (or procedural sky) is rendered
 * as the visible background.
 *
 * When `visible` is `true` (the default), the sky/env map is drawn behind all
 * geometry. In path-traced output, rays that escape the scene contribute the
 * full env/sky radiance. In the rasteriser preview, a fullscreen sky pass
 * renders the env map or procedural gradient before geometry is drawn.
 *
 * When `false`, the background shows a neutral dark grey instead — a "studio
 * lighting" look where the env map still illuminates the scene (via IBL) but
 * is not visible as the backdrop. Both renderers respect this flag.
 *
 * Path-tracer changes take effect on the next `render()` call; start a fresh
 * render (`sample_index = 0`) to avoid blending old and new samples.
 * Rasteriser changes take effect on the next `raster_frame()` call.
 */
export function set_env_background(visible: boolean): void;

/**
 * Sets the display exposure in EV (Exposure Value) stops.
 *
 * Applied as `linear_colour × 2^stops` just before ACES tone-mapping.
 * Because this is a post-accumulation transform, changing exposure does
 * **not** invalidate the accumulated samples — the running total stays
 * intact and the next `render` → `get_pixels` call will show the new
 * brightness. This makes the exposure slider the only control that can
 * update the live image mid-render without restarting.
 *
 * +1 EV = twice as bright; −1 EV = half as bright. Typical range: −3 to +3.
 */
export function set_exposure(stops: number): void;

/**
 * Sets the focal plane distance in world units.
 *
 * Geometry exactly this far from the camera appears sharp; everything closer
 * or farther blurs according to the aperture setting. Only meaningful when
 * `set_aperture` has been called with a value greater than zero. Changes take
 * effect on the next `render()` call with `sample_index = 0`.
 */
export function set_focus_distance(dist: number): void;

/**
 * Enables or disables Image-Based Lighting.
 *
 * When `enabled` is `true` (the default), the loaded HDR environment map is
 * used for ambient lighting and reflections in both the rasteriser preview and
 * the path tracer. When `false`, both renderers fall back to the procedural sky
 * gradient even if an env map is loaded — useful for A/B comparisons or artistic
 * control.
 *
 * Changes to the rasteriser preview take effect immediately on the next
 * `raster_frame()` call. Changes to the path tracer take effect on the next
 * `render()` call; start a fresh render (`sample_index = 0`) to avoid blending
 * old (IBL) and new (procedural sky) samples.
 */
export function set_ibl_enabled(enabled: boolean): void;

/**
 * Sets the IBL / sky brightness multiplier.
 *
 * `scale` multiplies all environment-map and procedural-sky contributions:
 * the visible background (when `env_background` is enabled), BRDF-sampled
 * escape rays, and the explicit env NEE shadow rays. It does **not** affect
 * emissive surfaces or the analytical sun.
 *
 * 0.0 = pitch-black environment; 1.0 = physical default; 2.0 = twice as
 * bright. Useful for quickly brightening a dim HDR map without editing
 * the file, or dimming the sky to match an indoor lighting scenario.
 */
export function set_ibl_scale(scale: number): void;

/**
 * Sets the maximum number of path-tracing bounces per sample.
 *
 * The value is clamped to `[2, 16]`: 2 is the minimum for any meaningful
 * indirect lighting (direct hit + one bounce), and 16 is enough to resolve
 * glass caustics and multi-room interiors without wasting GPU time on paths
 * that contribute almost nothing (Russian roulette handles the tail).
 *
 * Changes take effect on the next `render()` call. Start a new render
 * (`sample_index = 0`) to avoid blending old (high-bounce) and new
 * (low-bounce) samples together.
 */
export function set_max_bounces(n: number): void;

/**
 * Sets the sun azimuth angle.
 *
 * `degrees` is measured in the horizontal plane from the +Z axis toward
 * the +X axis (0° = due north / +Z, 90° = due east / +X). The shader
 * converts to a world-space direction at render time, so changing this
 * does not require re-uploading any buffers.
 *
 * Valid range: 0–360°. Values outside this range are accepted and wrap
 * correctly via trigonometry.
 */
export function set_sun_azimuth(degrees: number): void;

/**
 * Sets the sun elevation angle above the horizon.
 *
 * `degrees` = 0 puts the sun on the horizon; `degrees` = 90 puts it
 * directly overhead. Values are clamped by the shader's trig (a negative
 * elevation would just make the sun a below-horizon fill light, which is
 * unusual but physically plausible).
 *
 * Valid range: 0–90°.
 */
export function set_sun_elevation(degrees: number): void;

/**
 * Sets the sun intensity multiplier.
 *
 * `scale` directly multiplies `SUN_RADIANCE` in the shader's NEE
 * (Next Event Estimation) sun contribution. 0.0 effectively disables the
 * analytical sun; 1.0 is the physical default; values above 1.0 simulate
 * a brighter or harsher light source.
 *
 * Has no effect when an HDR environment map is active (the env map's own
 * sun is used instead via importance sampling).
 */
export function set_sun_intensity(scale: number): void;

/**
 * Removes the currently loaded HDR environment map and reverts to the
 * procedural sky gradient.
 *
 * After this call both the path tracer and the rasteriser preview behave as if
 * `load_environment` had never been called: background and IBL both come from
 * the procedural sky.
 *
 * This is a no-op if no environment map is currently loaded.
 */
export function unload_environment(): void;

/**
 * Sets the camera position and orientation and uploads the new uniform to
 * the GPU (both path-tracer and raster pipelines).
 *
 * - `px / py / pz` — world-space position
 * - `yaw`   — horizontal rotation in radians (0 = looking along −Z)
 * - `pitch` — vertical tilt in radians (positive = looking up)
 * - `vfov`  — vertical field of view in radians (e.g. π/3 for 60°)
 * - `aspect` — viewport width / height
 *
 * Call this every frame while the camera is moving (preview mode), or once
 * before kicking off a final render.
 */
export function update_camera(px: number, py: number, pz: number, yaw: number, pitch: number, vfov: number, aspect: number): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly get_pixels: (a: number, b: number) => any;
    readonly init_renderer: () => any;
    readonly load_environment: (a: number, b: number) => [number, number];
    readonly load_scene: (a: number, b: number) => [number, number];
    readonly raster_frame: (a: number, b: number) => void;
    readonly raster_get_pixels: (a: number, b: number) => any;
    readonly render: (a: number, b: number, c: number, d: number) => void;
    readonly set_aperture: (a: number) => void;
    readonly set_env_background: (a: number) => void;
    readonly set_exposure: (a: number) => void;
    readonly set_focus_distance: (a: number) => void;
    readonly set_ibl_enabled: (a: number) => void;
    readonly set_ibl_scale: (a: number) => void;
    readonly set_max_bounces: (a: number) => void;
    readonly set_sun_azimuth: (a: number) => void;
    readonly set_sun_elevation: (a: number) => void;
    readonly set_sun_intensity: (a: number) => void;
    readonly update_camera: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly get_scene_bounds: () => any;
    readonly unload_environment: () => void;
    readonly get_scene_stats: () => any;
    readonly wasm_bindgen__closure__destroy__h06d57fbbcf12cfb7: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__he5ef96efaea0f49b: (a: number, b: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h6b539ed7f51515d5: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h1522f83d29e2ee84: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h3d555e81212e6f69: (a: number, b: number, c: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
