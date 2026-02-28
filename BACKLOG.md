# Backlog

Items are listed roughly in order of priority within each section.
This file should be updated when work is started (move to a branch / PR)
and when items are completed (remove them and update README + CLAUDE.md).

---

## Interface Features

- **Adjustable sample count** — let the user choose render quality (e.g., 32 / 128 / 512 spp) instead of the hard-coded 256. A fast "preview quality" option would be useful for large canvases.
- **Save rendered image** — "Save PNG" button that exports the current canvas contents via `canvas.toBlob()`. Trivial to implement, high user value.
- **Camera FOV slider** — expose the vertical field of view so the user can switch between telephoto (narrow) and wide-angle looks without editing code.
- **Orbit pan mode** — hold Shift while dragging to translate the orbit target instead of rotating around it, so the user can reframe without zooming.
- **Camera reset button** — one click to re-run `autoFrame()` and return to the scene's default view.
- **Keyboard shortcuts** — R to start/cancel render, Escape to cancel, F to auto-frame, +/- for zoom. Show shortcuts in a help overlay.
- **Render region** — drag a rectangle on the canvas to path-trace only that region at full resolution; useful for examining a specific detail without a full-frame render.
- **Scene info panel** — show triangle count, mesh count, material count, and canvas resolution after a scene loads. Helps users understand what they've loaded.
- **Full-screen / maximise** — expand the canvas to fill the browser window with a single click.

---

## Rendering Features

- **MIS (Multiple Importance Sampling)** — balance samples between the BRDF lobe and the environment map using power heuristic weighting. Currently the firefly clamp suppresses bright BRDF hits; MIS eliminates the need for the clamp by giving rare bright paths appropriate probability.
- **Emissive mesh lights** — treat triangles with emissive materials as area light sources and sample them explicitly via NEE. Currently only the IBL and the analytical directional sun are sampled; adding mesh lights enables candles, lamps, and neon signs to illuminate scenes correctly.
- **Depth of field** — thin-lens model: jitter rays across a disc on the aperture and converge at a configurable focus distance. Adds cinematic bokeh with a focus distance slider in the UI.
- **Texture maps in the rasterizer** — the rasterizer currently uses solid `base_color_factor` only; sampling the `base_color_tex` array would make the interactive preview match the path-traced output much more closely.
- **Denoising** — integrate a denoising pass (either a WebGPU compute shader implementing a simple À-Trous or a port of OIDN) so that low sample counts (8–32 spp) produce a clean result. Would make the renderer much more interactive.
- **Adaptive sampling** — allocate more samples to high-variance regions (edges, specular highlights, caustics) and fewer to flat diffuse areas. Achieves a given perceived quality with fewer total samples.
- **Subsurface scattering** — approximate SSS for skin, wax, and foliage using a diffusion profile or a random-walk approximation. Required for realistic organic materials.
- **Volumetric scattering** — participating media (fog, smoke, clouds) via delta tracking through a homogeneous or heterogeneous medium. Needed for atmospheric and indoor scenes.
- **Anisotropic materials** — extend the GGX BRDF to support anisotropic roughness (e.g., brushed metal). GLTF 2.0 has the `KHR_materials_anisotropy` extension.
- **Thin-film interference** — iridescent coatings (soap bubbles, oil slicks, beetle wings) via the `KHR_materials_iridescence` extension.

---

## Performance Improvements

- **SAH BVH** — replace the current median-split build with Surface Area Heuristic splits. SAH trees typically halve traversal time on complex scenes at the cost of a slower build; BVH build time is already dominated by scene load, so the tradeoff is very favourable.
- **Two-level BVH (TLAS/BLAS)** — instead of flattening all instances into world-space triangles, keep per-mesh BLASes and build a TLAS over instance bounding boxes. Enables true instancing (one BLAS for many instances) and much smaller GPU memory for scenes with repeated geometry.
- **Russian roulette path termination** — terminate dim paths probabilistically instead of continuing them for a fixed bounce count. Unbiased (energy is conserved in expectation) and saves GPU time on paths that contribute little.
- **Wavefront path tracing** — reorganise the compute shader into a multi-pass wavefront where all active rays of the same type are processed together. Reduces thread divergence and improves GPU occupancy on scenes with mixed materials.
- **Texture compression** — upload textures in BC7 (desktop) or ETC2 (mobile) block-compressed format instead of RGBA8. Reduces GPU memory bandwidth and allows larger texture sets before hitting VRAM limits.
- **ReSTIR direct lighting** — Reservoir-based Spatio-Temporal Importance Resampling for direct illumination. Dramatic quality improvement at low spp on scenes with many emissive lights, at the cost of implementation complexity.
- **Tile-based rendering** — render in fixed-size tiles and stream results to the canvas progressively. Reduces peak VRAM use for very large canvas sizes.

---

## Maintenance

- **CI/CD pipeline** — GitHub Actions workflow that runs `cargo test`, `cargo clippy`, and the TypeScript tests (`npm test`) on every push and pull request. Prevents regressions from landing silently.
- **End-to-end browser tests** — Playwright tests that load the app in a headless browser, drop a known GLB, click Render, and compare the output image against a reference (pixel hash or structural similarity). Would have caught the scene-switch race condition automatically.
- **Benchmark suite** — a `cargo bench` / wall-clock harness that renders a fixed scene at a fixed sample count and records frame time. Run before and after performance changes to quantify improvements.
- **WASM source maps** — enable DWARF debug info in `wasm-pack` builds so that panics and crashes show readable Rust stack frames in browser DevTools rather than WASM bytecode offsets.
- **User-friendly error messages** — catch common WASM failure modes (scene too large for VRAM, unsupported GLTF extension, zero-triangle mesh) and surface them as readable status messages rather than raw Rust panic text.
- **Scene file validation** — check file size and magic bytes before passing to WASM; reject obviously wrong files with a helpful message before they trigger an opaque WASM error.
