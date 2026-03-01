# Backlog

Items are listed roughly in order of priority within each section.
This file should be updated when work is started (move to a branch / PR)
and when items are completed (remove them and update README + CLAUDE.md).

---

## Interface Features

- **Camera FOV slider** — expose the vertical field of view so the user can switch between telephoto (narrow) and wide-angle looks without editing code.
- **Orbit pan mode** — hold Shift while dragging to translate the orbit target instead of rotating around it, so the user can reframe without zooming.
- **Render region** — drag a rectangle on the canvas to path-trace only that region at full resolution; useful for examining a specific detail without a full-frame render.
- **Scene info panel** — show triangle count, mesh count, material count, and canvas resolution after a scene loads. Helps users understand what they've loaded.
- **Full-screen / maximise** — expand the canvas to fill the browser window with a single click.

---

## Renderer Controls (UI sliders and toggles in the Renderer tab)

Settings that let the user tune the renderer without editing code. Each item is a checkbox or slider wired to a constant or uniform already consumed by the shader.

- **Max bounce count slider** — let the user dial between 2 and 16 bounces. Low values (2–4) give fast, slightly-biased renders; high values (12–16) resolve multi-bounce glass caustics and dark interiors. Currently hardcoded to 12.
- **Sun direction controls** — two sliders (azimuth and elevation) to rotate the analytical sun. Essential for testing how a scene looks under different lighting angles without reloading or editing code.
- **Sun intensity slider** — multiply `SUN_RADIANCE` by a scalar. Lets the user dim or brighten sunlit regions relative to the IBL. Currently hardcoded to `vec3(10, 9, 7)`.
- **IBL intensity slider** — multiply every env-map sample by a scalar before it contributes to lighting. Lets the user balance IBL vs. the procedural sun without reloading the HDRI. Useful for scenes where the env map is too bright or too dark relative to the geometry.
- **Sky colour controls** — two colour pickers (or hue/lightness sliders) for `SKY_HORIZON` and `SKY_ZENITH`. Lets the user change the ambient colour cast of the procedural sky — warm sunset, cool overcast, alien green — without recompiling.
- **ACES exposure slider** — pre-multiply radiance by `2^stops` before tonemapping (range: −3 to +3 EV). Gives quick brightness control without resetting the accumulation buffer; useful when the scene is consistently over- or under-exposed.
- **Depth of field controls** — aperture (f-stop) and focus distance sliders, active once DoF is implemented. Listed here so the UI tab structure is ready when the rendering feature lands.

---

## Rendering Features

- **Emissive mesh lights** — treat triangles with emissive materials as area light sources and sample them explicitly via NEE. Currently only the IBL and the analytical directional sun are sampled; adding mesh lights enables candles, lamps, and neon signs to illuminate scenes correctly.
- **Depth of field** — thin-lens model: jitter rays across a disc on the aperture and converge at a configurable focus distance. Adds cinematic bokeh with a focus distance slider in the UI.
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
