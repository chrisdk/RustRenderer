/**
 * Built-in demo scenes, procedurally generated as GLB files.
 *
 * All geometry here is hand-authored and in the public domain. The Cornell
 * Box material values follow the reference measurement data published by
 * Cornell University's Program of Computer Graphics (freely available at
 * https://www.graphics.cornell.edu/online/box/data.html). No third-party
 * assets are used; no attribution is required.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

type Pt = [number, number, number];

/** A set of triangles that share a single material. */
interface Part {
    positions: number[];  // flat xyz triples, one entry per vertex
    normals:   number[];  // flat xyz triples, matching vertex count
    matIdx:    number;
}

/** GLTF PBR material description. */
interface GltfMaterial {
    baseColor: [number, number, number, number];
    roughness?: number;
    metallic?:  number;
    emissive?:  [number, number, number];
}

/** A named scene paired with a recommended starting camera. */
export interface BuiltinScene {
    readonly name:  string;
    readonly label: string;
    readonly camera: {
        position: [number, number, number];
        yaw:      number;
        pitch:    number;
    };
    build(): ArrayBuffer;
}

// ─────────────────────────────────────────────────────────────────────────────
// Geometry helpers
// ─────────────────────────────────────────────────────────────────────────────

function normalize3(v: Pt): Pt {
    const len = Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2);
    return [v[0] / len, v[1] / len, v[2] / len];
}

function cross3(a: Pt, b: Pt): Pt {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
}

function faceNormal(v0: Pt, v1: Pt, v2: Pt): Pt {
    const e1: Pt = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    const e2: Pt = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    return normalize3(cross3(e1, e2));
}

/**
 * Two CCW triangles forming a quad: (v0,v1,v2) + (v0,v2,v3).
 * Vertices must be wound counter-clockwise when viewed from the front face.
 * The face normal is computed from the first triangle and applied to all
 * six vertices (flat shading — our shader recomputes normals anyway).
 */
function quad(v0: Pt, v1: Pt, v2: Pt, v3: Pt, matIdx: number): Part {
    const n = faceNormal(v0, v1, v2);
    return {
        positions: [...v0, ...v1, ...v2, ...v0, ...v2, ...v3],
        normals:   [...n,  ...n,  ...n,  ...n,  ...n,  ...n],
        matIdx,
    };
}

/**
 * Five visible faces of an axis-aligned box (top, four sides; no bottom).
 * cx/cy/cz: centre. ex/ey/ez: half-extents.
 */
function box(
    cx: number, cy: number, cz: number,
    ex: number, ey: number, ez: number,
    matIdx: number,
): Part[] {
    const [x0, x1] = [cx - ex, cx + ex];
    const [y0, y1] = [cy - ey, cy + ey];
    const [z0, z1] = [cz - ez, cz + ez];
    return [
        quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], matIdx), // +Z
        quad([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], matIdx), // −Z
        quad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], matIdx), // −X
        quad([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], matIdx), // +X
        quad([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], matIdx), // +Y (top)
    ];
}

// ─────────────────────────────────────────────────────────────────────────────
// GLB packer
// ─────────────────────────────────────────────────────────────────────────────

function minMax3(floats: number[]): { min: number[]; max: number[] } {
    let [mnX, mnY, mnZ] = [ Infinity,  Infinity,  Infinity];
    let [mxX, mxY, mxZ] = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < floats.length; i += 3) {
        mnX = Math.min(mnX, floats[i]);     mxX = Math.max(mxX, floats[i]);
        mnY = Math.min(mnY, floats[i + 1]); mxY = Math.max(mxY, floats[i + 1]);
        mnZ = Math.min(mnZ, floats[i + 2]); mxZ = Math.max(mxZ, floats[i + 2]);
    }
    return { min: [mnX, mnY, mnZ], max: [mxX, mxY, mxZ] };
}

/**
 * Pack an array of Parts and a material list into a binary GLTF (GLB) file.
 *
 * Binary layout:
 *   bufferView 0 — all position arrays concatenated
 *   bufferView 1 — all normal   arrays concatenated
 *
 * Each part gets one mesh with one TRIANGLES primitive, referencing its own
 * POSITION + NORMAL accessors and material.
 */
function buildGlb(parts: Part[], materials: GltfMaterial[]): ArrayBuffer {
    // Flatten all positions and normals.
    const allPos  = parts.flatMap(p => p.positions);
    const allNorm = parts.flatMap(p => p.normals);

    const posBytes  = allPos.length  * 4;
    const normBytes = allNorm.length * 4;

    const binData = new Float32Array(allPos.length + allNorm.length);
    binData.set(allPos,  0);
    binData.set(allNorm, allPos.length);

    // Build per-part accessors (two per part: POSITION then NORMAL).
    const accessors: object[] = [];
    let posOff  = 0;
    let normOff = 0;

    for (const part of parts) {
        const count = part.positions.length / 3;
        const posMM  = minMax3(part.positions);
        const normMM = minMax3(part.normals);

        accessors.push({ bufferView: 0, byteOffset: posOff,  componentType: 5126,
                         count, type: 'VEC3', min: posMM.min,  max: posMM.max  });
        accessors.push({ bufferView: 1, byteOffset: normOff, componentType: 5126,
                         count, type: 'VEC3', min: normMM.min, max: normMM.max });

        posOff  += count * 12;
        normOff += count * 12;
    }

    const meshes = parts.map((p, i) => ({
        primitives: [{
            attributes: { POSITION: i * 2, NORMAL: i * 2 + 1 },
            material: p.matIdx,
            mode: 4,  // TRIANGLES
        }],
    }));

    const gltfMats = materials.map(m => {
        const entry: Record<string, unknown> = {
            pbrMetallicRoughness: {
                baseColorFactor: m.baseColor,
                roughnessFactor: m.roughness ?? 0.9,
                metallicFactor:  m.metallic  ?? 0.0,
            },
        };
        if (m.emissive) entry['emissiveFactor'] = m.emissive;
        return entry;
    });

    const json = {
        asset:  { version: '2.0' },
        scene:  0,
        scenes: [{ nodes: parts.map((_, i) => i) }],
        nodes:  parts.map((_, i) => ({ mesh: i })),
        meshes,
        materials: gltfMats,
        accessors,
        bufferViews: [
            { buffer: 0, byteOffset: 0,        byteLength: posBytes  },
            { buffer: 0, byteOffset: posBytes,  byteLength: normBytes },
        ],
        buffers: [{ byteLength: posBytes + normBytes }],
    };

    return packGlb(json, binData);
}

/** Serialise a GLTF JSON object + binary blob into a GLB ArrayBuffer. */
function packGlb(json: object, binData: Float32Array): ArrayBuffer {
    const jsonStr = JSON.stringify(json);
    const jsonPad = (4 - (jsonStr.length        % 4)) % 4;
    const binPad  = (4 - (binData.byteLength    % 4)) % 4;
    const jsonLen = jsonStr.length + jsonPad;
    const binLen  = binData.byteLength + binPad;
    const total   = 12 + 8 + jsonLen + 8 + binLen;

    const buf  = new ArrayBuffer(total);
    const dv   = new DataView(buf);
    const u8   = new Uint8Array(buf);
    let   off  = 0;

    // File header
    dv.setUint32(off, 0x46546C67, true); off += 4;  // 'glTF'
    dv.setUint32(off, 2,          true); off += 4;  // version 2
    dv.setUint32(off, total,      true); off += 4;

    // JSON chunk — pad with ASCII spaces (required by the GLB spec)
    dv.setUint32(off, jsonLen,     true); off += 4;
    dv.setUint32(off, 0x4E4F534A, true); off += 4;  // 'JSON'
    u8.set(new TextEncoder().encode(jsonStr), off);
    u8.fill(0x20, off + jsonStr.length, off + jsonLen);
    off += jsonLen;

    // Binary chunk — pad with zeros (required by the GLB spec)
    dv.setUint32(off, binLen,      true); off += 4;
    dv.setUint32(off, 0x004E4942, true); off += 4;  // 'BIN\0'
    u8.set(new Uint8Array(binData.buffer), off);
    // trailing zeros already present from ArrayBuffer zero-initialisation

    return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene definitions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The Cornell Box — the canonical path-tracing test scene since 1984.
 *
 * A closed room with a white floor, ceiling, and back wall; a red left wall;
 * a green right wall; and a small warm-light panel just below the ceiling.
 * Material values approximate the physical measurements from Cornell's
 * original radiosity paper (Goral et al., SIGGRAPH 1984).
 */
function cornellBox(): ArrayBuffer {
    const materials: GltfMaterial[] = [
        { baseColor: [0.73, 0.73, 0.73, 1] },                       // 0: white
        { baseColor: [0.65, 0.05, 0.05, 1] },                       // 1: red
        { baseColor: [0.12, 0.45, 0.15, 1] },                       // 2: green
        { baseColor: [1.00, 0.90, 0.70, 1], emissive: [1, 0.9, 0.7] }, // 3: warm light
    ];

    // Each quad uses CCW winding so its computed face normal points inward
    // (toward the interior of the box, where the rays will hit from).
    const parts: Part[] = [
        quad([-1,-1, 1], [ 1,-1, 1], [ 1,-1,-1], [-1,-1,-1], 0),  // floor   (+Y)
        quad([-1, 1,-1], [ 1, 1,-1], [ 1, 1, 1], [-1, 1, 1], 0),  // ceiling (−Y)
        quad([-1,-1,-1], [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1], 0),  // back    (+Z)
        quad([-1,-1,-1], [-1, 1,-1], [-1, 1, 1], [-1,-1, 1], 1),  // left    (+X) red
        quad([ 1,-1, 1], [ 1, 1, 1], [ 1, 1,-1], [ 1,-1,-1], 2),  // right   (−X) green
        // Light panel slightly below y=+1 so the ceiling quad doesn't z-fight it.
        quad([-0.40, 0.998,-0.40], [ 0.40, 0.998,-0.40],
             [ 0.40, 0.998, 0.40], [-0.40, 0.998, 0.40], 3),       // light   (−Y)
    ];

    return buildGlb(parts, materials);
}

/**
 * Three coloured boxes on a stone ground plane.
 *
 * A quick sanity check for the renderer under open-sky conditions: tests
 * material colours, the directional light, and multiple objects casting
 * (well, not yet casting — that's Phase 4) shadows on each other.
 */
function openScene(): ArrayBuffer {
    const materials: GltfMaterial[] = [
        { baseColor: [0.42, 0.40, 0.35, 1] },  // 0: stone ground
        { baseColor: [0.80, 0.18, 0.15, 1] },  // 1: red
        { baseColor: [0.15, 0.30, 0.80, 1] },  // 2: blue
        { baseColor: [0.85, 0.68, 0.10, 1] },  // 3: gold
    ];

    // Box helper: cy is the CENTRE y, so cy=ey places the bottom face at y=0.
    const parts: Part[] = [
        quad([-10, 0, 10], [10, 0, 10], [10, 0, -10], [-10, 0, -10], 0), // ground
        ...box(-2.5, 0.60, -3.5, 0.60, 0.60, 0.60, 1),  // red    cube
        ...box( 0.0, 1.00, -4.5, 0.50, 1.00, 0.50, 2),  // blue   tower
        ...box( 2.5, 0.45, -3.5, 0.80, 0.45, 0.80, 3),  // gold   slab
    ];

    return buildGlb(parts, materials);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public exports
// ─────────────────────────────────────────────────────────────────────────────

export const BUILTIN_SCENES: readonly BuiltinScene[] = [
    {
        name:  'cornell-box',
        label: 'Cornell Box',
        // Camera sits just inside the open front face, centred vertically,
        // so the box fills the frame at 60° vfov.
        camera: { position: [0, 0, 0.75], yaw: 0, pitch: 0 },
        build:  cornellBox,
    },
    {
        name:  'open-scene',
        label: 'Open Scene',
        camera: { position: [0, 1.5, 6], yaw: 0, pitch: -0.1 },
        build:  openScene,
    },
];
