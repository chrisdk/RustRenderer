# 3D Rendering App

A physically-based path tracer that runs in the browser. The rendering engine is written in Rust, compiled to WebAssembly, and driven by a TypeScript frontend using WebGPU.

## Stack
- **Frontend**: TypeScript (not JavaScript), no framework, no bundler. Communicates with the engine via WASM.
- **Rendering Engine**: Rust, compiled to WASM via `wasm-pack`. GPU work done through WebGPU (`wgpu` crate).
- **Scene format**: GLTF / GLB, loaded with the `gltf = "1"` crate.
- **Math**: `glam = "0.29"` for matrix/vector ops (used during scene loading and BVH build).
- **LSP**: `rust-analyzer-lsp` is installed on this machine. Use it for type checking, finding definitions, and getting diagnostics on Rust code without waiting for a full `cargo build`.

## How it works (big picture)

Two rendering modes:

1. **Interactive preview** — while the user drags or zooms the camera, a hardware rasterizer (vertex + fragment shader) redraws every frame entirely on the GPU. No CPU readback, no path tracing — stable 60 fps even on complex scenes.
2. **Final render** — triggered by clicking "Render"; the path tracer accumulates samples progressively until cancelled. Automatically resets on camera move.

The WASM API (in `src/lib.rs`) exposes:
1. `load_scene(bytes)` — parse GLTF, build GPU buffers, build BVH, upload to both renderers.
2. `update_camera(...)` — update the camera uniform buffer; resets path-tracer accumulation.
3. `render()` / `get_pixels()` — one path-tracer sample accumulation pass + staging readback.
4. `raster_frame(width, height)` / `raster_get_pixels(width, height)` — hardware rasterizer frame.

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
src/camera.rs            Camera + Ray + RasterCameraUniform; ray generation, camera navigation
src/renderer/
  mod.rs                 Re-exports Renderer, HitRecord, intersect_bvh
  gpu.rs                 Renderer struct — path-tracer pipeline, scene storage buffers
  raster.rs              RasterRenderer — hardware rasterizer pipeline + instance draw list
  math.rs                dot, cross, reflect, fresnel_schlick
  intersect.rs           ray_aabb (slab method) + ray_triangle (Möller–Trumbore)
  traverse.rs            intersect_bvh — iterative BVH traversal, returns HitRecord
src/shaders/
  trace.wgsl             Full PBR path-tracer compute shader (GGX, IBL, NEE, ACES)
  raster.wgsl            PBR-approximation vertex + fragment shaders for interactive preview
www/
  src/index.ts           Loads WASM, wires canvas + turntable camera + controls
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

**`Camera`** (`src/camera.rs`) — updated interactively as the user navigates; separate from the static Scene.
- `position: [f32; 3]`, `yaw: f32`, `pitch: f32`, `vfov: f32`, `aspect: f32`
- `ray(u, v) -> Ray` — generates a world-space ray for normalized pixel coords; (0,0) = top-left, (1,1) = bottom-right
- `pan(dyaw, dpitch)` — mouse-look; pitch is clamped to ±(π/2 − 0.001) to prevent flipping
- Basis vectors (forward, right, up) are derived analytically from yaw/pitch — no drift, no quaternions
- The frontend uses a **turntable orbit** model: position is computed from (azimuth, elevation, radius) in TypeScript and passed to the WASM `update_camera` call; the Rust camera struct itself is position+yaw+pitch only.

**`RasterCameraUniform`** (`src/camera.rs`):
- 128-byte uniform: `view: [[f32;4];4]` + `proj: [[f32;4];4]` — column-major
- Built by `Camera::to_raster_uniform(aspect)` from the same analytical basis vectors as the path tracer, so both modes see identical geometry.

**`Ray`** (`src/camera.rs`):
- `origin: [f32; 3]`, `direction: [f32; 3]` — direction is always normalized

## Key architectural decisions

- **Two rendering pipelines, one scene.** The rasterizer and path tracer share the same `Scene` data (vertex/index buffers, materials, textures). Uploading once in `load_scene` feeds both.
- **Geometry stays in local space.** Transforms are stored per-instance and applied at BVH build time. This enables instancing without duplicating vertex data.
- **Flat vertex + index buffers.** All meshes share one `Vec<Vertex>` and one `Vec<u32>`. A `Mesh` is just a (first_index, index_count) window into those buffers. GPU-friendly; no per-mesh overhead.
- **RGBA8 everywhere.** All textures are normalized to RGBA8 at load time regardless of source format. The GPU never has to handle multiple texture formats.
- **Pre-order BVH layout.** Left child is always at `node_idx + 1`, so only the right child index needs explicit storage. This matches how GPU traversal stacks work.
- **Matrices are column-major `[[f32; 4]; 4]`.** This matches GLTF, glam, and WGSL conventions. `m[col][row]`.
- **Camera basis is derived analytically.** Forward/right/up are computed from yaw and pitch on demand rather than stored. Avoids accumulated floating-point drift from repeated matrix multiplications.
- **NaN guards before accumulation.** The path tracer explicitly sanitises each sample's radiance (`if any(r != r) { r = 0; }`) before adding it to the accumulation buffer. One NaN in the buffer would permanently corrupt that pixel (NaN + x = NaN forever due to IEEE 754).

## What's built

- [x] GLTF scene loader (`Scene::from_gltf`) — geometry, materials, textures, scene graph traversal, instancing
- [x] BVH construction (`Bvh::build`) — median split on longest axis, flat pre-order layout
- [x] Camera (`Camera`, `Ray`) — ray generation, analytical basis vectors, turntable orbit in frontend
- [x] CPU renderer (`src/renderer/`) — math helpers, ray–AABB (slab), ray–triangle (Möller–Trumbore), iterative BVH traversal
- [x] GPU path tracer — `trace.wgsl` compute shader: full PBR GGX, metallic/roughness BRDF, glass/transmission, normal maps, emissive surfaces
- [x] IBL (Image-Based Lighting) — HDRI environment maps, equirectangular sampling, firefly suppression
- [x] NEE (Next Event Estimation) — analytical directional sun light sampled explicitly each bounce
- [x] Progressive accumulation — running weighted average of samples; ACES tonemapping on display
- [x] Hardware rasterizer — `raster.rs` / `raster.wgsl`: PBR approximation, per-instance transforms, 60 fps interactive preview
- [x] Turntable camera — drag-to-orbit, scroll-to-zoom (mouse wheel + touchpad pinch), auto-framing on scene load
- [x] Portability — crate builds as both `cdylib` (WASM) and `rlib` (native); WASM glue is `#[cfg(wasm32)]`-gated throughout
- [x] WASM API — `init_renderer`, `load_scene`, `update_camera`, `render`, `get_pixels`, `raster_frame`, `raster_get_pixels`
- [x] TypeScript frontend — scene picker, drag-and-drop GLTF/HDR loader, render button, HQ/preview mode switching
- 91+ unit tests, all passing (`cargo test`)

## What's next

1. **MIS (Multiple Importance Sampling)** — proper way to sample IBL without needing the firefly clamp; balance between BRDF sampling and environment map sampling
2. **Texture maps in the rasterizer** — currently uses solid `base_color_factor` only; sampling the full texture array would make the preview match the path-tracer output better
3. **Depth of field** — thin-lens model: sample a disc on the aperture and converge at a focus distance
4. **Emissive mesh lights** — area lights from geometry with emissive materials; currently only IBL + analytical sun

## License

MIT — see `LICENSE`.

## Code style

### Comments
Comment code thoroughly. Assume the reader is not an expert in rendering or graphics programming. Explain:
- What a type or function does and why it exists, not just what its fields/parameters are named.
- Any non-obvious algorithm, formula, or data layout decision.
- Rendering concepts (BVH, path tracing, PBR, etc.) the first time they appear in a file.

Comments must stay accurate as code evolves. An outdated comment is worse than no comment — it actively misleads the next reader. When you change code, update the surrounding comments in the same commit.

Use `///` doc comments on all public items. Use `//` inline comments for anything that would not be immediately obvious to a competent Rust developer unfamiliar with rendering.

### Personality
Be a little entertaining. A well-placed joke in a comment, a touch of wit in a doc string, a wry observation about why ray tracing is simultaneously beautiful and computationally horrifying — all welcome. Keep it professional enough to live in a codebase, but fun enough that reading the source isn't a chore.

### Testing
Write tests for all non-trivial logic. This means:
- Every mathematical operation with a known closed-form answer gets a unit test.
- Every data structure invariant (BVH layout, buffer alignment, GPU struct sizes) gets an assertion test — use `assert_eq!(std::mem::size_of::<Foo>(), N)` to catch silent ABI breaks.
- Edge cases matter: degenerate geometry (zero-area triangles), missing GLTF fields, out-of-range parameters.
- When a bug is fixed, add a regression test so it stays fixed.

Run `cargo test` before committing. If the test suite is broken, fix it before doing anything else. A green test suite is a prerequisite, not an afterthought.

### README
Keep `README.md` up to date whenever features are added, changed, or removed. The README is the front door; if it describes a feature that no longer exists (or omits one that does), it misleads everyone who reads it. Update the Status section when a planned feature ships.

### Commits
Write meaningful commit messages. The subject line should complete the sentence "This commit will…". Use the body to explain *why* the change was made, not just what files were touched. Keep subject lines under 72 characters.
