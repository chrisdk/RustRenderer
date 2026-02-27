/**
 * Built-in demo scenes sourced from the Khronos glTF Sample Assets collection.
 * https://github.com/KhronosGroup/glTF-Sample-Assets
 *
 * No assets are bundled — files are fetched from raw.githubusercontent.com on
 * first use and cached by the browser for repeat visits.
 *
 * Licensing:
 *   Most models are CC0 (public domain). The Damaged Helmet is CC BY 4.0 and
 *   requires attribution — see the `attribution` field on that entry.
 */

const KHRONOS_BASE =
    'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models';

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
     * Attribution string required by the asset's licence.
     * Present only on non-CC0 assets. The frontend displays this below the
     * scene button so credit is always visible to the user.
     */
    readonly attribution?: string;
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

/** Shorthand: build a fetchGlb call for a Khronos sample model. */
function khronos(model: string): () => Promise<ArrayBuffer> {
    return () => fetchGlb(`${KHRONOS_BASE}/${model}/glTF-Binary/${model}.glb`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public exports
// ─────────────────────────────────────────────────────────────────────────────

export const BUILTIN_SCENES: readonly BuiltinScene[] = [
    // ── CC BY 4.0 ─────────────────────────────────────────────────────────────
    {
        name:        'damaged-helmet',
        label:       'Damaged Helmet',
        // CC BY 4.0 — attribution required.
        // Original by ctxwing on Sketchfab; adapted for glTF by theblueturtle_.
        attribution: '© ctxwing / theblueturtle_ · CC BY 4.0',
        build:       khronos('DamagedHelmet'),
    },

    {
        name:        'glass-hurricane-candle-holder',
        label:       'Glass Hurricane Candle Holder',
        // CC BY 4.0 — attribution required.
        // Created by Eric Chadwick for Wayfair, LLC (2021).
        attribution: '© 2021 Wayfair / Eric Chadwick · CC BY 4.0',
        build:       khronos('GlassHurricaneCandleHolder'),
    },

    // ── CC0 (public domain) ───────────────────────────────────────────────────
    {
        name:  'water-bottle',
        label: 'Water Bottle',
        build: khronos('WaterBottle'),
    },
    {
        name:  'lantern',
        label: 'Lantern',
        build: khronos('Lantern'),
    },
    {
        name:  'boom-box',
        label: 'BoomBox',
        build: khronos('BoomBox'),
    },
    {
        name:  'antique-camera',
        label: 'Antique Camera',
        build: khronos('AntiqueCamera'),
    },
    {
        name:  'toy-car',
        label: 'Toy Car',
        build: khronos('ToyCar'),
    },
];
