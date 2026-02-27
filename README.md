# RustRenderer

A physically-based path tracer that runs in the browser. The rendering engine is written in Rust, compiled to WebAssembly, and driven by a TypeScript frontend using WebGPU.

## What it does

Load a GLTF scene, navigate it with FPS-style camera controls, then trigger a high-quality still render when you have the shot you want. Both the interactive preview and the final render use the same path tracing algorithm on the GPU — the preview just uses fewer samples per pixel so it stays fast.

Path tracing produces realistic light transport: soft shadows, indirect illumination, reflections, and area lights all emerge naturally from the algorithm rather than being special-cased.

## Stack

| Layer | Technology |
|---|---|
| Rendering engine | Rust → WebAssembly (`wasm-pack`) |
| GPU | WebGPU (via `wgpu`) |
| Frontend | TypeScript (no framework, no bundler) |
| Scene format | GLTF / GLB |

## Project structure

```
src/              Rust rendering engine
  lib.rs          WASM entry point; exports init_renderer / load_scene / render
  accel/          Acceleration structures
    bvh.rs        Bounding Volume Hierarchy — built once at scene load time
  scene/          Scene representation (loaded from GLTF)
    mod.rs        Scene struct + GLTF loader (two-pass: geometry then scene graph)
    geometry.rs   Vertex, Mesh, MeshInstance
    material.rs   PBR material (metallic/roughness workflow)
    texture.rs    Texture (all formats normalized to RGBA8)
    environment.rs  HDRI environment map + tint
  camera.rs       Camera + Ray; ray generation and FPS-style navigation
  renderer/       GPU renderer
    gpu.rs        Renderer struct — device, queue, scene storage buffers
    math.rs       Vector math helpers (dot, cross, reflect, Fresnel)
    intersect.rs  Ray–AABB and ray–triangle intersection (slab + Möller–Trumbore)
    traverse.rs   Iterative BVH traversal; produces a HitRecord per ray
www/              TypeScript frontend
  src/index.ts    Loads the WASM module, wires up the canvas and controls
  index.html      Single-page app shell
  tsconfig.json
benches/          Criterion benchmarks
build.sh          Build script (wasm-pack + tsc)
run.sh            Build and serve locally
```

## How the renderer works

### Scene loading

Scenes are loaded from GLTF files (`.gltf` or self-contained `.glb` bundles). Loading happens in two passes:

1. **Geometry pass** — reads every mesh in the GLTF mesh library into flat vertex and index buffers. Each mesh is stored in its own local coordinate space.
2. **Scene graph pass** — traverses the GLTF node tree, accumulating parent transforms as it goes. Every node that references a mesh produces a `MeshInstance` carrying the full world-space transform.

Keeping geometry in local space and recording transforms separately enables instancing: a scene with 100 identical chairs stores the chair geometry once and 100 transforms, not 100 copies of the geometry.

### Acceleration structure

Before rendering, the BVH (Bounding Volume Hierarchy) is built from the scene:

- Every mesh instance is expanded into world-space triangles (applying its transform).
- The triangles are recursively divided into a binary tree by sorting along the longest axis and splitting at the median.
- The tree is flattened into a compact array of `BvhNode` structs (32 bytes each) for efficient GPU upload.

During rendering, each ray traverses this tree to find intersections in O(log n) time rather than testing every triangle.

### Rendering

The path tracer runs as a WebGPU compute shader. Each pixel accumulates multiple samples; each sample traces a ray from the camera through that pixel, bouncing around the scene until it hits a light or escapes. The average of all samples converges to the correct image.

- **Preview mode** — 1–4 samples per pixel, updated continuously as the camera moves.
- **Final render mode** — many samples accumulated progressively until the image converges.

## Building

**Prerequisites:** `rustup`, `wasm-pack`, `tsc` (TypeScript compiler), `python3`.

```bash
# Install wasm-pack if you don't have it
cargo install wasm-pack

# Build WASM + TypeScript
./build.sh
```

## Running locally

```bash
./run.sh
# Open http://localhost:8080
```

`run.sh` builds the project then serves the `www/` directory with Python's built-in HTTP server.

## Sample scenes

GLTF sample scenes are available from the [Khronos GLTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets) repository. The renderer accepts both `.gltf` (JSON + separate binary) and `.glb` (self-contained binary bundle) files.

## Status

Early development. What's done:

- [x] GLTF scene loading (geometry, materials, textures, scene graph / instancing)
- [x] BVH construction (median split, flat pre-order layout)
- [x] Camera (ray generation, FPS pan/translate, analytical basis vectors)
- [x] CPU renderer foundation (math, ray–AABB, ray–triangle, BVH traversal — 68 tests)
- [x] GPU setup (wgpu device/queue, scene storage buffer upload)
- [x] Portability: core is frontend-agnostic; WASM glue is isolated to `lib.rs`

In progress / planned:

- [ ] Camera uniform buffer + WASM API (`load_scene`, `update_camera`, `render`)
- [ ] WGSL path tracing compute shader (port of the CPU traversal)
- [ ] Compute pipeline, bind groups, dispatch
- [ ] Frontend controls (FPS-style WASD + mouse-look, render trigger button)
