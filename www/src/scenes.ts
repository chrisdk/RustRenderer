/**
 * Built-in demo scenes backed by remote CC0 GLB assets from the Khronos
 * glTF Sample Assets collection.
 *
 * No assets are bundled — files are fetched from raw.githubusercontent.com on
 * first use and cached by the browser for repeat visits.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** A named scene paired with an optional recommended starting camera. */
export interface BuiltinScene {
    readonly name:  string;
    readonly label: string;
    /**
     * Explicit camera preset for scenes that need a specific viewpoint (e.g.
     * inside a Cornell Box). When absent the frontend auto-frames the scene
     * from its bounding box so the model fills ~80% of the viewport.
     */
    readonly camera?: {
        position: [number, number, number];
        yaw:      number;
        pitch:    number;
    };
    /**
     * Returns the GLB bytes for this scene.
     * Remote scenes return a Promise<ArrayBuffer> that fetches the file.
     * Call sites should always await Promise.resolve(scene.build()).
     */
    build(): ArrayBuffer | Promise<ArrayBuffer>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Remote scene loader
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fetch a remote GLB file and return its bytes.
 *
 * We rely on the browser's HTTP cache for repeat visits — raw.githubusercontent.com
 * sends sensible cache headers, so a re-click after the first download is fast.
 */
async function fetchGlb(url: string): Promise<ArrayBuffer> {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);
    return resp.arrayBuffer();
}

// ─────────────────────────────────────────────────────────────────────────────
// Public exports
// ─────────────────────────────────────────────────────────────────────────────

// CC0 (public domain). No attribution legally required, but thanks Khronos!
export const BUILTIN_SCENES: readonly BuiltinScene[] = [
    {
        name:  'water-bottle',
        label: 'Water Bottle',
        build: () => fetchGlb(
            'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets' +
            '/main/Models/WaterBottle/glTF-Binary/WaterBottle.glb',
        ),
    },
    {
        name:  'lantern',
        label: 'Lantern',
        build: () => fetchGlb(
            'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets' +
            '/main/Models/Lantern/glTF-Binary/Lantern.glb',
        ),
    },
];
