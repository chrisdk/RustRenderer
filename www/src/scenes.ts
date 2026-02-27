/**
 * Built-in demo scenes, procedurally generated as GLB files.
 *
 * All geometry here is hand-authored and in the public domain.
 *
 * Cornell Box geometry and material values follow the reference measurements
 * published by Cornell University's Program of Computer Graphics (available at
 * https://www.graphics.cornell.edu/online/box/data.html).
 *
 * No third-party assets are used; no attribution is required.
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
 * six vertices (flat shading).
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
 * Five visible faces of an axis-aligned box (top + four sides; no bottom).
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

/**
 * UV sphere centred at (cx, cy, cz) with the given radius.
 *
 * Uses per-vertex outward normals rather than flat face normals, so the
 * geometry is ready for smooth shading when the renderer supports it.
 *
 * `stacks` = latitude bands (rings from pole to pole).
 * `slices` = longitude segments (slices around the equator).
 * More of each → rounder sphere, more triangles. 20×32 is a good default.
 *
 * Winding verification (outward-normal CCW rule):
 *   For any quad on the sphere surface, viewing from outside, vertices go
 *   v00 → v01 → v11 → v10 (bottom-left, top-left, top-right, bottom-right in
 *   angular space), which produces normals pointing away from the centre.
 */
function uvSphere(
    cx: number, cy: number, cz: number, r: number,
    stacks: number, slices: number,
    matIdx: number,
): Part {
    const positions: number[] = [];
    const normals:   number[] = [];

    for (let i = 0; i < stacks; i++) {
        const phi0 = Math.PI * i / stacks       - Math.PI / 2;
        const phi1 = Math.PI * (i + 1) / stacks - Math.PI / 2;

        for (let j = 0; j < slices; j++) {
            const theta0 = 2 * Math.PI * j       / slices;
            const theta1 = 2 * Math.PI * (j + 1) / slices;

            // Outward unit normal at a (phi, theta) point on the unit sphere.
            // X = east, Y = up, Z = south (matches our world coordinate system).
            const nrm = (phi: number, theta: number): Pt => [
                Math.cos(phi) * Math.cos(theta),
                Math.sin(phi),
                Math.cos(phi) * Math.sin(theta),
            ];

            // Four corners of this lat/lon quad.
            const n00 = nrm(phi0, theta0);
            const n10 = nrm(phi0, theta1);
            const n11 = nrm(phi1, theta1);
            const n01 = nrm(phi1, theta0);

            const vtx = (n: Pt): Pt => [cx + r * n[0], cy + r * n[1], cz + r * n[2]];
            const v00 = vtx(n00), v10 = vtx(n10), v11 = vtx(n11), v01 = vtx(n01);

            // Two CCW triangles: (v00, v01, v11) + (v00, v11, v10).
            // Produces outward-pointing face normals (verified analytically).
            positions.push(...v00, ...v01, ...v11,  ...v00, ...v11, ...v10);
            normals.push(  ...n00, ...n01, ...n11,  ...n00, ...n11, ...n10);
        }
    }

    return { positions, normals, matIdx };
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
    const allPos  = parts.flatMap(p => p.positions);
    const allNorm = parts.flatMap(p => p.normals);

    const posBytes  = allPos.length  * 4;
    const normBytes = allNorm.length * 4;

    const binData = new Float32Array(allPos.length + allNorm.length);
    binData.set(allPos,  0);
    binData.set(allNorm, allPos.length);

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
            mode: 4,
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

    dv.setUint32(off, 0x46546C67, true); off += 4;
    dv.setUint32(off, 2,          true); off += 4;
    dv.setUint32(off, total,      true); off += 4;

    dv.setUint32(off, jsonLen,     true); off += 4;
    dv.setUint32(off, 0x4E4F534A, true); off += 4;
    u8.set(new TextEncoder().encode(jsonStr), off);
    u8.fill(0x20, off + jsonStr.length, off + jsonLen);
    off += jsonLen;

    dv.setUint32(off, binLen,      true); off += 4;
    dv.setUint32(off, 0x004E4942, true); off += 4;
    u8.set(new Uint8Array(binData.buffer), off);

    return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene definitions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The Cornell Box — the canonical path-tracing reference scene since 1984.
 *
 * A closed room with a white floor, ceiling, and back wall; a red left wall;
 * a green right wall; and a warm area light just below the ceiling. Two
 * white blocks occupy the lower half of the room, as in the original paper
 * by Goral et al. (SIGGRAPH 1984).
 *
 * Material values approximate the physical spectral measurements from:
 *   https://www.graphics.cornell.edu/online/box/data.html
 *
 * This scene is the acid test for any global illumination renderer: the
 * characteristic red and green colour bleeding onto the white surfaces only
 * appears with indirect illumination. It looks fine with direct lighting and
 * spectacular once multi-bounce path tracing is in.
 */
function cornellBox(): ArrayBuffer {
    const materials: GltfMaterial[] = [
        { baseColor: [0.73, 0.73, 0.73, 1] },                            // 0: white
        { baseColor: [0.65, 0.05, 0.05, 1] },                            // 1: red
        { baseColor: [0.12, 0.45, 0.15, 1] },                            // 2: green
        { baseColor: [1.00, 0.90, 0.70, 1], emissive: [1, 0.9, 0.7] },  // 3: warm area light
    ];

    // Room walls — CCW winding so face normals point inward (toward the rays).
    const parts: Part[] = [
        quad([-1,-1, 1], [ 1,-1, 1], [ 1,-1,-1], [-1,-1,-1], 0),  // floor
        quad([-1, 1,-1], [ 1, 1,-1], [ 1, 1, 1], [-1, 1, 1], 0),  // ceiling
        quad([-1,-1,-1], [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1], 0),  // back wall
        quad([-1,-1,-1], [-1, 1,-1], [-1, 1, 1], [-1,-1, 1], 1),  // left wall  (red)
        quad([ 1,-1, 1], [ 1, 1, 1], [ 1, 1,-1], [ 1,-1,-1], 2),  // right wall (green)

        // Area light: a panel flush with the ceiling, emitting downward.
        quad([-0.40, 0.998,-0.40], [ 0.40, 0.998,-0.40],
             [ 0.40, 0.998, 0.40], [-0.40, 0.998, 0.40], 3),

        // The two classic blocks from the original paper. Approximated at
        // their canonical proportions: a tall thin block on the left and a
        // shorter wider block on the right.
        ...box(-0.35, -0.40, -0.20, 0.27, 0.60, 0.27, 0),  // tall block (left)
        ...box( 0.35, -0.70,  0.25, 0.27, 0.30, 0.27, 0),  // short block (right)
    ];

    return buildGlb(parts, materials);
}

/**
 * Cornell Spheres — the same room with two spheres instead of blocks.
 *
 * A classic variant of the Cornell Box that gives the path tracer more
 * interesting geometry to work with. The large white sphere on the left and
 * the smaller gold sphere on the right produce very different shading under
 * indirect illumination: the white sphere picks up red and green colour
 * bleeding from the walls; the gold sphere adds warm reflective highlights
 * once the PBR shader is in.
 *
 * Camera position and room dimensions are identical to the standard Cornell
 * Box so the two can be compared directly.
 */
function cornellSpheres(): ArrayBuffer {
    const materials: GltfMaterial[] = [
        { baseColor: [0.73, 0.73, 0.73, 1] },                            // 0: white
        { baseColor: [0.65, 0.05, 0.05, 1] },                            // 1: red
        { baseColor: [0.12, 0.45, 0.15, 1] },                            // 2: green
        { baseColor: [1.00, 0.90, 0.70, 1], emissive: [1, 0.9, 0.7] },  // 3: warm area light
        { baseColor: [0.73, 0.73, 0.73, 1], roughness: 0.0 },            // 4: mirror sphere
        { baseColor: [0.83, 0.60, 0.10, 1], roughness: 0.1, metallic: 1.0 }, // 5: gold sphere
    ];

    const parts: Part[] = [
        // Room
        quad([-1,-1, 1], [ 1,-1, 1], [ 1,-1,-1], [-1,-1,-1], 0),  // floor
        quad([-1, 1,-1], [ 1, 1,-1], [ 1, 1, 1], [-1, 1, 1], 0),  // ceiling
        quad([-1,-1,-1], [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1], 0),  // back wall
        quad([-1,-1,-1], [-1, 1,-1], [-1, 1, 1], [-1,-1, 1], 1),  // left wall  (red)
        quad([ 1,-1, 1], [ 1, 1, 1], [ 1, 1,-1], [ 1,-1,-1], 2),  // right wall (green)
        quad([-0.40, 0.998,-0.40], [ 0.40, 0.998,-0.40],
             [ 0.40, 0.998, 0.40], [-0.40, 0.998, 0.40], 3),       // area light

        // Spheres.
        // 20×32 gives a smooth silhouette without excessive triangle count.
        uvSphere(-0.38, -0.55, -0.10, 0.42, 20, 32, 4),  // large white sphere (left)
        uvSphere( 0.37, -0.72,  0.25, 0.26, 20, 32, 5),  // small gold sphere  (right)
    ];

    return buildGlb(parts, materials);
}

/**
 * Neon Grotto — a dark cave lit entirely by floating emissive orbs.
 *
 * Four glowing spheres — warm orange, electric blue, acid green, and pale
 * gold — are the only light sources. Under the current single-bounce shader
 * they look like luminous baubles in the dark. Under full path tracing each
 * orb will cast soft coloured light onto the floor and walls, and the colours
 * will mix where the spheres' illumination overlaps.
 *
 * The scene is deliberately dark and high-contrast so that the transition from
 * direct-only to full global illumination is as dramatic as possible.
 */
function neonGrotto(): ArrayBuffer {
    // Near-black walls; just enough albedo to show off the geometry.
    const cave  = 0.025;
    const materials: GltfMaterial[] = [
        { baseColor: [cave, cave, cave * 1.2, 1] },  // 0: dark grey-blue cave walls
        { baseColor: [1.0, 0.40, 0.05, 1], emissive: [1.0, 0.35, 0.02] },  // 1: orange orb
        { baseColor: [0.1, 0.30, 1.00, 1], emissive: [0.0, 0.20, 1.00] },  // 2: blue orb
        { baseColor: [0.1, 1.00, 0.20, 1], emissive: [0.0, 0.90, 0.10] },  // 3: green orb
        { baseColor: [1.0, 0.95, 0.60, 1], emissive: [1.0, 0.85, 0.40] },  // 4: gold orb
    ];

    // A low-ceilinged rectangular cave: wider than it is tall.
    // x ∈ [−2, 2], y ∈ [−1, 0.7], z ∈ [−2.5, 0.5]
    const [x0, x1] = [-2.0, 2.0];
    const [y0, y1] = [-1.0, 0.7];
    const [z0, z1] = [-2.5, 0.5];

    const parts: Part[] = [
        // Cave walls — all six faces, inward-facing normals.
        quad([x0,y0,z1], [x1,y0,z1], [x1,y1,z1], [x0,y1,z1], 0),  // front wall
        quad([x1,y0,z0], [x0,y0,z0], [x0,y1,z0], [x1,y1,z0], 0),  // back wall
        quad([x0,y0,z0], [x0,y0,z1], [x0,y1,z1], [x0,y1,z0], 0),  // left wall
        quad([x1,y0,z1], [x1,y0,z0], [x1,y1,z0], [x1,y1,z1], 0),  // right wall
        quad([x0,y0,z1], [x0,y0,z0], [x1,y0,z0], [x1,y0,z1], 0),  // floor (up-facing)
        quad([x0,y1,z0], [x0,y1,z1], [x1,y1,z1], [x1,y1,z0], 0),  // ceiling (down-facing)

        // Four emissive orbs at different depths and heights.
        // They're staggered so each one is clearly distinct from the others
        // and their coloured light pools don't fully overlap.
        uvSphere(-1.0, -0.25, -1.8, 0.30, 20, 32, 1),  // orange — back left,  mid-height
        uvSphere( 0.9, -0.55, -1.0, 0.25, 20, 32, 2),  // blue   — centre-right, low
        uvSphere(-0.4,  0.10, -0.4, 0.22, 20, 32, 3),  // green  — near,         high
        uvSphere( 1.2, -0.15, -2.0, 0.28, 20, 32, 4),  // gold   — back right,   mid
    ];

    return buildGlb(parts, materials);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public exports
// ─────────────────────────────────────────────────────────────────────────────

export const BUILTIN_SCENES: readonly BuiltinScene[] = [
    {
        name:   'cornell-box',
        label:  'Cornell Box',
        // Camera sits just inside the open front face, centred vertically,
        // so the box fills the frame at 60° vfov.
        camera: { position: [0, 0, 0.75], yaw: 0, pitch: 0 },
        build:  cornellBox,
    },
    {
        name:   'cornell-spheres',
        label:  'Cornell Spheres',
        camera: { position: [0, 0, 0.75], yaw: 0, pitch: 0 },
        build:  cornellSpheres,
    },
    {
        name:   'neon-grotto',
        label:  'Neon Grotto',
        // Positioned near the front opening, looking down-and-in so all four
        // orbs are visible in the frame.
        camera: { position: [0, 0.3, 1.2], yaw: 0, pitch: -0.18 },
        build:  neonGrotto,
    },
];
