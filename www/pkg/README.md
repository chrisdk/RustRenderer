# RustRenderer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A physically-based path tracer that runs in the browser. The rendering engine is written in Rust, compiled to WebAssembly, and driven by a TypeScript frontend using WebGPU.

## What it does

Load a GLTF scene, orbit it with a turntable camera (drag to spin, scroll to zoom), then click **Render** for a high-quality path-traced still when you have the shot you want.

Two rendering modes, one scene:

- **Interactive preview** — while you drag or zoom, a hardware rasterizer (vertex + fragment shaders) runs entirely on the GPU for stable 60 fps. No path tracing, no CPU readback, no lag.
- **Final render** — the path tracer takes over: multiple bounces, PBR materials, image-based lighting from an HDRI environment, and progressive sample accumulation until the image converges or you cancel.

Path tracing produces realistic light transport — soft shadows, indirect illumination, reflections, glass refraction — all emerging naturally from the algorithm rather than being special-cased.

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
  lib.rs          WASM entry point; exports load_scene / render / raster_frame / …
  accel/
    bvh.rs        Bounding Volume Hierarchy — built once at scene load
  scene/
    mod.rs        Scene struct + GLTF loader (two-pass: geometry then scene graph)
    geometry.rs   Vertex, Mesh, MeshInstance
    material.rs   PBR material (metallic/roughness workflow)
    texture.rs    Texture (all formats normalised to RGBA8)
    environment.rs  HDRI environment map + tint
  camera.rs       Camera + Ray + RasterCameraUniform; ray generation, basis vectors
  renderer/
    gpu.rs        Renderer — path-tracer pipeline, scene storage buffers
    raster.rs     RasterRenderer — hardware rasterizer, per-instance draw list
    math.rs       Vector math helpers (dot, cross, reflect, Fresnel)
    intersect.rs  Ray–AABB (slab) and ray–triangle (Möller–Trumbore) intersection
    traverse.rs   Iterative BVH traversal; returns a HitRecord per ray
  shaders/
    trace.wgsl    Full PBR path-tracer compute shader (GGX, IBL, NEE, ACES)
    raster.wgsl   PBR-approximation vertex + fragment shaders (60 fps preview)
www/              TypeScript frontend
  src/index.ts    Loads WASM, wires canvas, turntable camera, controls
  index.html      Single-page app shell
  tsconfig.json
benches/          Criterion benchmarks
build.sh          Build script (wasm-pack + tsc)
run.sh            Build and serve locally
```

## How the renderer works

### Scene loading

Scenes are loaded from GLTF files (`.gltf` or self-contained `.glb` bundles). Loading happens in two passes:

1. **Geometry pass** — reads every mesh into flat vertex and index buffers. Each mesh stays in its own local coordinate space.
2. **Scene graph pass** — traverses the GLTF node tree, accumulating parent transforms. Every node that references a mesh produces a `MeshInstance` carrying the full world-space transform.

Keeping geometry local and recording transforms separately enables instancing: a scene with 100 identical chairs stores the geometry once and 100 transforms, not 100 copies.

Both the rasterizer and the path tracer share the same uploaded scene data; loading once feeds both pipelines.

### Acceleration structure

Before path tracing, a BVH (Bounding Volume Hierarchy) is built:

- Mesh instances are expanded into world-space triangles.
- Triangles are recursively divided into a binary tree by splitting at the median along the longest axis.
- The tree is flattened into compact `BvhNode` structs (32 bytes each) for GPU upload.

During rendering each ray traverses the tree to find the nearest intersection in O(log n) time.

### Interactive preview (rasterizer)

While the user manipulates the camera a hardware rasterizer produces each frame using a standard vertex + fragment shader pipeline. It applies per-instance transforms and a Phong/PBR approximation with the same materials as the path tracer. Because the output goes directly to a GPU texture (no CPU readback), it stays at 60 fps regardless of scene complexity.

### Final render (path tracer)

The path tracer runs as a WebGPU compute shader (`trace.wgsl`). Each pixel accumulates multiple samples; each sample bounces a ray around the scene:

- **GGX microfacet BRDF** — physically correct specular reflection using the GGX normal distribution and Smith's shadowing-masking term.
- **Metallic/roughness workflow** — full PBR: conductor (metallic) surfaces use the full BRDF; dielectric surfaces separate diffuse and specular lobes.
- **Glass / transmission** — Fresnel-weighted refraction using Snell's law, with TIR (total internal reflection) handled correctly.
- **Normal maps** — tangent-space normal maps applied via a tangent-space TBN matrix; UV-seam NaN guards prevent permanent pixel corruption.
- **Image-based lighting (IBL)** — HDRI equirectangular environment maps used for both the path-tracer background and indirect light. Load a custom `.hdr` file or use the default.
- **Next Event Estimation (NEE)** — the sun direction is sampled explicitly at each bounce, dramatically reducing variance on outdoor scenes.
- **Progressive accumulation** — each call to `render()` adds one more sample per pixel. Samples are averaged and displayed with ACES filmic tonemapping.
- **Firefly suppression** — radiance is clamped per-sample before accumulation (threshold tuned to ACES saturation) to prevent single bright samples burning in permanently.

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

GLTF sample scenes are available from the [Khronos glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets) repository. The renderer accepts both `.gltf` (JSON + separate binary) and `.glb` (self-contained binary bundle) files. Drop a file directly onto the canvas or use **Load GLB / GLTF…** in the panel.

For environment lighting, load any equirectangular `.hdr` file via **Load HDR environment…**.

## Status

91+ unit tests passing (`cargo test`).

### Done

- [x] GLTF scene loading — geometry, materials, textures, scene graph traversal, instancing
- [x] BVH construction — median split on longest axis, flat pre-order layout
- [x] Full PBR GGX path tracer — metallic/roughness BRDF, glass/transmission, normal maps, emissive
- [x] IBL — HDRI environment maps with equirectangular sampling and firefly suppression
- [x] NEE — analytical directional sun sampled explicitly at each bounce
- [x] Progressive multi-sample accumulation with ACES filmic tonemapping
- [x] Hardware rasterizer — 60 fps interactive preview with PBR approximation
- [x] Turntable camera — drag-to-orbit, scroll-to-zoom (mouse wheel + touchpad pinch), auto-framing
- [x] Drag-and-drop scene and environment loading
- [x] Portability — core builds as `cdylib` (WASM) and `rlib` (native tests)

### Planned

- [ ] MIS (Multiple Importance Sampling) — balance BRDF and IBL sampling for faster convergence
- [ ] Texture maps in rasterizer — currently uses solid base colour only in preview mode
- [ ] Depth of field — thin-lens model with configurable aperture and focus distance
- [ ] Emissive mesh lights — area lights from geometry; currently only IBL + sun

## License

MIT — see [LICENSE](LICENSE).
