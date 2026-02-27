//! GPU state: device, queue, and scene-data storage buffers.
//!
//! [`Renderer`] owns every piece of GPU state needed to trace a frame.
//! It is constructed once at startup (async, because WebGPU adapter and
//! device acquisition go through promise-based browser APIs), then reused
//! across every frame.
//!
//! ## Lifecycle
//!
//! ```text
//! Renderer::new()           — pick adapter → open device + queue
//!     ↓
//! renderer.upload_scene()   — BVH + materials → GPU storage buffers
//!     ↓
//! renderer.render_frame()   — dispatch compute shader  (Phase 3)
//! ```
//!
//! ## Why storage buffers?
//!
//! Path tracing requires random access to the full BVH and material array
//! from every shader invocation. Storage buffers (`@storage`) give the WGSL
//! shader unrestricted read access by index — unlike uniform buffers, which
//! have a tight 64 KiB size limit that a real scene would blow past instantly.

use crate::accel::bvh::Bvh;
use crate::scene::material::Material;

// ============================================================================
// Public types
// ============================================================================

/// All GPU state required to trace rays through a scene.
///
/// Fields that depend on a loaded scene (`bvh_nodes_buf`, etc.) are `None`
/// until [`Renderer::upload_scene`] is called. Attempting to render before
/// that will be a no-op (or an error, once the render pipeline is wired up
/// in Phase 3).
pub struct Renderer {
    /// The logical GPU device: allocates buffers, creates pipelines, records
    /// command encoders. Think of it as a handle to the GPU driver.
    pub(crate) device: wgpu::Device,

    /// The GPU command queue: buffers up command encoders and submits them
    /// to the GPU for execution. Every `queue.submit(...)` call is one GPU
    /// dispatch.
    pub(crate) queue: wgpu::Queue,

    /// Flat pre-order array of BVH nodes on the GPU.
    ///
    /// WGSL binding: `@group(0) @binding(0) var<storage, read> bvh_nodes`.
    /// Layout: see [`BvhNode`](crate::accel::bvh::BvhNode) — 32 bytes each.
    pub(crate) bvh_nodes_buf: Option<wgpu::Buffer>,

    /// World-space triangles reordered to match BVH leaf order, on the GPU.
    ///
    /// WGSL binding: `@group(0) @binding(1) var<storage, read> bvh_tris`.
    /// Layout: see [`BvhTriangle`](crate::accel::bvh::BvhTriangle) — 64 bytes each.
    pub(crate) bvh_tris_buf: Option<wgpu::Buffer>,

    /// PBR materials, indexed by `BvhTriangle::material_index`, on the GPU.
    ///
    /// WGSL binding: `@group(0) @binding(2) var<storage, read> materials`.
    /// Layout: see [`Material`](crate::scene::material::Material) — 64 bytes each.
    pub(crate) materials_buf: Option<wgpu::Buffer>,
}

// ============================================================================
// Implementation
// ============================================================================

impl Renderer {
    /// Selects a GPU adapter and opens a logical device on it.
    ///
    /// This function is `async` because WebGPU exposes hardware selection
    /// through browser promises (`navigator.gpu.requestAdapter()` etc.),
    /// which surface in Rust/WASM as `Future`s. On native targets (e.g.
    /// `cargo test`) wgpu drives the async internally.
    ///
    /// # Errors
    ///
    /// Returns a human-readable error string if:
    /// - No suitable adapter is found — the browser doesn't support WebGPU,
    ///   the machine has no GPU, or we're running in a headless CI environment
    ///   without a software rasterizer.
    /// - The device request fails — driver crash, out of VRAM, etc.
    pub async fn new() -> Result<Self, String> {
        // `Instance` is the wgpu entry point: it enumerates physical adapters
        // and is the factory for surfaces. One instance per process is typical.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            // `all()` selects the best available backend at runtime:
            //   - WebGPU in a modern browser (wasm32 target)
            //   - Vulkan / Metal / DX12 in a native test binary
            // This keeps the native test path and the browser production path
            // on the same codegen, so bugs found in tests are real bugs.
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // An `Adapter` is a handle to a specific physical GPU (or a software
        // rasterizer). `HighPerformance` prefers a discrete dGPU when the
        // system has both integrated and discrete options — exactly what you
        // want when you're about to do thousands of ray bounces per frame.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                "no suitable GPU adapter found — is WebGPU supported?".to_string()
            })?;

        let info = adapter.get_info();
        log::info!("GPU adapter: {} ({:?})", info.name, info.backend);

        // Open a logical `Device` (resource allocator + pipeline factory) and
        // its associated `Queue` (command submission). These are the two
        // objects you touch for every GPU operation.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("path tracer"),
                    // No exotic feature requirements for now — compute shaders
                    // are available on every WebGPU-capable device.
                    ..Default::default()
                },
                None, // no wgpu API-call trace file
            )
            .await
            .map_err(|e| format!("GPU device creation failed: {e}"))?;

        Ok(Self {
            device,
            queue,
            bvh_nodes_buf: None,
            bvh_tris_buf:  None,
            materials_buf: None,
        })
    }

    /// Uploads BVH geometry and materials into GPU storage buffers.
    ///
    /// After this call the three scene buffers are populated and the renderer
    /// is ready to have a compute pipeline bound and dispatched. The CPU-side
    /// `bvh` and `materials` data can be dropped once this returns — the GPU
    /// holds its own copy.
    ///
    /// Calling this a second time (e.g. after loading a new scene) silently
    /// drops the old buffers and creates fresh ones.
    ///
    /// ## Binding layout
    ///
    /// | Buffer        | Binding | WGSL type                |
    /// |---------------|---------|--------------------------|
    /// | bvh_nodes     | 0       | `array<BvhNode>`         |
    /// | bvh_triangles | 1       | `array<BvhTriangle>`     |
    /// | materials     | 2       | `array<Material>`        |
    ///
    /// The WGSL struct definitions must byte-for-byte match the Rust types
    /// (enforced by the layout tests in `accel::bvh` and `scene::material`).
    pub fn upload_scene(&mut self, bvh: &Bvh, materials: &[Material]) {
        use wgpu::util::DeviceExt;

        // `STORAGE` exposes the buffer to shader code as a storage buffer.
        // `COPY_DST` allows `queue.write_buffer()` updates later — handy for
        // animated geometry or swapping scenes without rebuilding the pipeline.
        let flags = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        // `create_buffer_init` allocates the buffer AND uploads the initial
        // data in one call. `bytemuck::cast_slice` is the zero-copy spell that
        // turns `&[BvhNode]` into `&[u8]` without any intermediate allocation —
        // possible because BvhNode is #[repr(C)] + Pod.
        self.bvh_nodes_buf = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label:    Some("bvh_nodes"),
                contents: bytemuck::cast_slice(&bvh.nodes),
                usage:    flags,
            },
        ));

        self.bvh_tris_buf = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label:    Some("bvh_triangles"),
                contents: bytemuck::cast_slice(&bvh.triangles),
                usage:    flags,
            },
        ));

        self.materials_buf = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label:    Some("materials"),
                contents: bytemuck::cast_slice(materials),
                usage:    flags,
            },
        ));

        log::info!(
            "scene uploaded: {} BVH nodes, {} triangles, {} materials",
            bvh.nodes.len(),
            bvh.triangles.len(),
            materials.len(),
        );
    }

    /// Returns `true` once [`upload_scene`] has been called and the scene
    /// buffers are live on the GPU. The compute pipeline must not be
    /// dispatched until this returns `true`.
    pub fn is_scene_loaded(&self) -> bool {
        self.bvh_nodes_buf.is_some()
    }
}
