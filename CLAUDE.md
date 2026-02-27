# 3D Rendering App

A physically-based path tracer that runs in the browser. The rendering engine is written in Rust, compiled to WebAssembly, and driven by a TypeScript frontend using WebGPU.

## Stack
- **Frontend**: TypeScript (not JavaScript), no framework, no bundler. Communicates with the engine via WASM.
- **Rendering Engine**: Rust, compiled to WASM via `wasm-pack`. GPU work done through WebGPU (`wgpu` crate).
- **Scene format**: GLTF / GLB, loaded with the `gltf = "1"` crate.
- **Math**: `glam = "0.29"` for matrix/vector ops (used during scene loading and BVH build).

## How it works (big picture)

Two rendering modes, same algorithm:
1. **Preview** — user navigates the scene FPS-style; renders at 1–4 samples per pixel so it stays interactive.
2. **Final render** — triggered explicitly; accumulates many samples until the image converges.

The WASM API (in `src/lib.rs`) exposes three steps:
1. `load_scene(bytes)` — parse GLTF, build GPU buffers, build BVH.
2. `update_camera(...)` — update the camera, trigger a preview render.
3. `render()` — kick off a high-quality accumulation render.

## Module map

```
src/lib.rs               WASM entry point; #[wasm_bindgen] exports
src/scene/
  mod.rs                 Scene struct + from_gltf() loader
  geometry.rs            Vertex, Mesh, MeshInstance
  material.rs            Material (PBR metallic/roughness)
  texture.rs             Texture (all formats normalized to RGBA8)
  environment.rs         Environment (HDRI map + tint)
src/accel/
  mod.rs                 Re-exports Bvh
  bvh.rs                 BVH build + BvhNode / BvhTriangle types
www/
  src/index.ts           Loads WASM, wires canvas + controls
  index.html             App shell
  tsconfig.json
build.sh                 wasm-pack + tsc
run.sh                   build.sh then python3 HTTP server on :8080
```

## Key types

**`Scene`** (`src/scene/mod.rs`) — owns everything loaded from a GLTF file. Immutable after load.
- `vertices: Vec<Vertex>` — flat buffer, all meshes concatenated
- `indices: Vec<u32>` — flat buffer, every 3 entries = one triangle
- `meshes: Vec<Mesh>` — ranges into the flat buffers (first_index, index_count, material_index)
- `instances: Vec<MeshInstance>` — (mesh_index, world_transform) pairs; same mesh can appear many times
- `materials: Vec<Material>`, `textures: Vec<Texture>`, `environment: Option<Environment>`

**`Bvh`** (`src/accel/bvh.rs`) — built from a `&Scene` by `Bvh::build()`. Ready for GPU upload.
- `nodes: Vec<BvhNode>` — flat pre-order array; left child always at `node_idx + 1`, right child stored explicitly in `right_or_first`
- `triangles: Vec<BvhTriangle>` — world-space triangles reordered to match BVH leaf order

**`BvhNode`** (32 bytes, GPU-friendly):
- `aabb_min: [f32; 3]`, `aabb_max: [f32; 3]`
- `right_or_first: u32` — internal: right child index; leaf: first triangle index
- `count: u32` — 0 = internal node; >0 = leaf, triangle count

## Key architectural decisions

- **Geometry stays in local space.** Transforms are stored per-instance and applied at BVH build time. This enables instancing without duplicating vertex data.
- **Flat vertex + index buffers.** All meshes share one `Vec<Vertex>` and one `Vec<u32>`. A `Mesh` is just a (first_index, index_count) window into those buffers. GPU-friendly; no per-mesh overhead.
- **RGBA8 everywhere.** All textures are normalized to RGBA8 at load time regardless of source format. The GPU never has to handle multiple texture formats.
- **Pre-order BVH layout.** Left child is always at `node_idx + 1`, so only the right child index needs explicit storage. This matches how GPU traversal stacks work.
- **Matrices are column-major `[[f32; 4]; 4]`.** This matches GLTF, glam, and WGSL conventions. `m[col][row]`.

## What's built

- [x] GLTF scene loader (`Scene::from_gltf`) — geometry, materials, textures, scene graph traversal, instancing
- [x] BVH construction (`Bvh::build`) — median split on longest axis, flat pre-order layout
- [x] Unit tests for BVH (7 tests: empty scene, single triangle, AABB containment, leaf coverage, tree invariants, transforms, instancing)

## What's next

1. **Camera** (`src/camera.rs`) — ray generation from position + orientation; updated interactively by the frontend
2. **WebGPU pipeline** (`src/renderer/`) — device setup, buffer uploads, WGSL compute shader for path tracing
3. **Frontend controls** — camera navigation (FPS-style), render trigger button

## Code style

### Comments
Comment code thoroughly. Assume the reader is not an expert in rendering or graphics programming. Explain:
- What a type or function does and why it exists, not just what its fields/parameters are named.
- Any non-obvious algorithm, formula, or data layout decision.
- Rendering concepts (BVH, path tracing, PBR, etc.) the first time they appear in a file.

Use `///` doc comments on all public items. Use `//` inline comments for anything that would not be immediately obvious to a competent Rust developer unfamiliar with rendering.

### Personality
Be a little entertaining. A well-placed joke in a comment, a touch of wit in a doc string, a wry observation about why ray tracing is simultaneously beautiful and computationally horrifying — all welcome. Keep it professional enough to live in a codebase, but fun enough that reading the source isn't a chore.

### Commits
Write meaningful commit messages. The subject line should complete the sentence "This commit will…". Use the body to explain *why* the change was made, not just what files were touched. Keep subject lines under 72 characters.
