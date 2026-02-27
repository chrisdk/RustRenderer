//! GPU state: device, queue, pipelines, and scene/frame buffers.
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
//! renderer.upload_camera()  — camera uniform → GPU uniform buffer
//!     ↓
//! renderer.render_frame()   — dispatch compute shader → writes output_buf
//!     ↓
//! renderer.get_pixels()     — copy output_buf → staging → Vec<u8>
//! ```
//!
//! ## Buffer / binding layout
//!
//! | Group | Binding | Name         | Type    | Usage              |
//! |-------|---------|--------------|---------|---------------------|
//! | 0     | 0       | bvh_nodes    | storage | read               |
//! | 0     | 1       | bvh_tris     | storage | read               |
//! | 0     | 2       | materials    | storage | read               |
//! | 1     | 0       | camera       | uniform | read               |
//! | 1     | 1       | frame        | uniform | read               |
//! | 1     | 2       | output       | storage | read_write         |

use crate::accel::bvh::Bvh;
use crate::camera::CameraUniform;
use crate::scene::material::Material;

// ============================================================================
// Supporting types
// ============================================================================

/// Per-frame shader constants: output dimensions and the sample index (reserved
/// for progressive accumulation in Phase 4).
///
/// Must be 16-byte aligned for a WGSL uniform buffer; the `_pad` field
/// satisfies the 16-byte minimum size.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrameUniforms {
    pub width:        u32,
    pub height:       u32,
    pub sample_index: u32,
    pub _pad:         u32,
}

// ============================================================================
// Renderer
// ============================================================================

/// All GPU state required to trace rays through a scene.
pub struct Renderer {
    /// The logical GPU device: allocates buffers and creates pipelines.
    pub(crate) device: wgpu::Device,
    /// The GPU command queue: submits encoded command buffers for execution.
    pub(crate) queue: wgpu::Queue,

    // ── Scene data (uploaded once per loaded scene) ──────────────────────────
    pub(crate) bvh_nodes_buf: Option<wgpu::Buffer>,
    pub(crate) bvh_tris_buf:  Option<wgpu::Buffer>,
    pub(crate) materials_buf: Option<wgpu::Buffer>,

    // ── Per-frame data (updated before each dispatch) ────────────────────────
    /// Camera parameters. `@group(1) @binding(0)`.
    pub(crate) camera_buf: Option<wgpu::Buffer>,
    /// Frame dimensions + sample index. `@group(1) @binding(1)`.
    pub(crate) frame_buf:  Option<wgpu::Buffer>,
    /// Output pixel buffer (RGBA8). `@group(1) @binding(2)`.
    pub(crate) output_buf: Option<wgpu::Buffer>,
    /// Dimensions of the current output buffer; used to detect resizes.
    pub(crate) output_dims: (u32, u32),

    // ── Pipeline (built lazily on first render_frame) ────────────────────────
    pub(crate) scene_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) frame_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) pipeline:  Option<wgpu::ComputePipeline>,
}

// ============================================================================
// Construction
// ============================================================================

impl Renderer {
    /// Selects a GPU adapter and opens a logical device on it.
    ///
    /// This function is `async` because WebGPU exposes hardware selection
    /// through browser promises (`navigator.gpu.requestAdapter()` etc.),
    /// which surface in Rust/WASM as `Future`s. On native targets wgpu
    /// drives the async internally.
    ///
    /// # Errors
    ///
    /// Returns a human-readable error string if no suitable adapter is found
    /// or if device creation fails.
    pub async fn new() -> Result<Self, String> {
        // wgpu 28: Instance::new takes &InstanceDescriptor (borrowed).
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // wgpu 28: request_adapter returns Result<Adapter, E>, not Option<Adapter>.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("no suitable GPU adapter found: {e}"))?;

        let info = adapter.get_info();
        log::info!("GPU adapter: {} ({:?})", info.name, info.backend);

        // wgpu 28: request_device takes only &DeviceDescriptor (trace-path arg removed).
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("path tracer"),
                    ..Default::default()
                },
            )
            .await
            .map_err(|e| format!("GPU device creation failed: {e}"))?;

        Ok(Self {
            device,
            queue,
            bvh_nodes_buf: None,
            bvh_tris_buf:  None,
            materials_buf: None,
            camera_buf:    None,
            frame_buf:     None,
            output_buf:    None,
            output_dims:   (0, 0),
            scene_bgl:     None,
            frame_bgl:     None,
            pipeline:      None,
        })
    }
}

// ============================================================================
// Scene and camera uploads
// ============================================================================

impl Renderer {
    /// Uploads BVH geometry and materials into GPU storage buffers.
    ///
    /// Calling this a second time drops the old buffers and creates fresh ones.
    pub fn upload_scene(&mut self, bvh: &Bvh, materials: &[Material]) {
        use wgpu::util::DeviceExt;
        let flags = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

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
            bvh.nodes.len(), bvh.triangles.len(), materials.len(),
        );
    }

    /// Uploads (or updates) the camera uniform buffer.
    ///
    /// On the first call the buffer is allocated and filled. On every subsequent
    /// call `queue.write_buffer` updates it in-place — no allocation, no stall.
    pub fn upload_camera(&mut self, uniform: &CameraUniform) {
        let data = bytemuck::bytes_of(uniform);
        match &self.camera_buf {
            Some(buf) => self.queue.write_buffer(buf, 0, data),
            None => {
                use wgpu::util::DeviceExt;
                self.camera_buf = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label:    Some("camera_uniform"),
                        contents: data,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    },
                ));
            }
        }
        log::debug!("camera uploaded: pos={:?}", uniform.position);
    }

    /// Returns `true` once [`upload_scene`] has been called.
    pub fn is_scene_loaded(&self) -> bool {
        self.bvh_nodes_buf.is_some()
    }
}

// ============================================================================
// Pipeline construction (lazy, called on first render_frame)
// ============================================================================

impl Renderer {
    fn build_pipeline(&mut self) {
        // ── Shader ──────────────────────────────────────────────────────────
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("trace.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/trace.wgsl").into()
            ),
        });

        // ── Bind group layouts ───────────────────────────────────────────────
        // Group 0: scene data — three read-only storage buffers.
        let scene_bgl = self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("scene_bgl"),
                entries: &[
                    storage_ro(0), // bvh_nodes
                    storage_ro(1), // bvh_tris
                    storage_ro(2), // materials
                ],
            }
        );

        // Group 1: per-frame data — two uniforms + one read-write storage.
        let frame_bgl = self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("frame_bgl"),
                entries: &[
                    uniform(0),     // camera
                    uniform(1),     // frame (width, height, sample_index)
                    storage_rw(2),  // output pixel buffer
                ],
            }
        );

        // ── Pipeline layout and compute pipeline ─────────────────────────────
        let layout = self.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label:                Some("trace_layout"),
                bind_group_layouts:   &[&scene_bgl, &frame_bgl],
                // wgpu 28: push_constant_ranges removed; replaced by immediate_size.
                immediate_size: 0,
            }
        );

        let pipeline = self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label:       Some("trace_pipeline"),
                layout:      Some(&layout),
                module:      &shader,
                // wgpu 28: entry_point is Option<&str> again.
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache:       None,
            }
        );

        self.scene_bgl = Some(scene_bgl);
        self.frame_bgl = Some(frame_bgl);
        self.pipeline  = Some(pipeline);

        log::info!("compute pipeline ready");
    }
}

// ── Helpers for building bind group layout entries ───────────────────────────

fn storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    }
}

fn storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    }
}

fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    }
}

// ============================================================================
// Rendering
// ============================================================================

impl Renderer {
    /// Dispatches the compute shader for one sample pass.
    ///
    /// Builds the pipeline on the first call (shader compilation happens here).
    /// The output is written to an internal `output_buf`; call [`get_pixels`]
    /// afterwards to read the result back to the CPU.
    ///
    /// Panics if [`upload_scene`] or [`upload_camera`] has not been called.
    pub fn render_frame(&mut self, width: u32, height: u32) {
        assert!(self.is_scene_loaded(), "render_frame called before upload_scene");
        assert!(self.camera_buf.is_some(), "render_frame called before upload_camera");

        // Lazy pipeline build — compiles the WGSL shader on first call.
        if self.pipeline.is_none() {
            self.build_pipeline();
        }

        // (Re)create the output buffer when dimensions change.
        let pixel_count = (width * height) as u64;
        let output_size = pixel_count * 4; // 4 bytes per RGBA8 pixel
        if self.output_dims != (width, height) {
            self.output_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("output"),
                size:               output_size,
                usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.output_dims = (width, height);
        }

        // Upload frame uniforms.
        let frame_data = FrameUniforms { width, height, sample_index: 0, _pad: 0 };
        match &self.frame_buf {
            Some(buf) => self.queue.write_buffer(buf, 0, bytemuck::bytes_of(&frame_data)),
            None => {
                use wgpu::util::DeviceExt;
                self.frame_buf = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label:    Some("frame_uniform"),
                        contents: bytemuck::bytes_of(&frame_data),
                        usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    },
                ));
            }
        }

        // Build bind groups (cheap descriptor objects; fine to recreate per frame).
        let scene_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("scene_bg"),
            layout:  self.scene_bgl.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bvh_nodes_buf.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bvh_tris_buf.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.materials_buf.as_ref().unwrap().as_entire_binding() },
            ],
        });

        let frame_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("frame_bg"),
            layout:  self.frame_bgl.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.camera_buf.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.frame_buf.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.output_buf.as_ref().unwrap().as_entire_binding() },
            ],
        });

        // Encode and submit the compute pass.
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("trace") }
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("trace_pass"), timestamp_writes: None }
            );
            pass.set_pipeline(self.pipeline.as_ref().unwrap());
            pass.set_bind_group(0, &scene_bg, &[]);
            pass.set_bind_group(1, &frame_bg, &[]);
            // Workgroup size is 8×8; dispatch enough groups to cover every pixel.
            pass.dispatch_workgroups(
                (width  + 7) / 8,
                (height + 7) / 8,
                1,
            );
        }
        self.queue.submit(Some(encoder.finish()));
        log::debug!("render_frame dispatched ({}×{})", width, height);
    }

    /// Copies the rendered pixels from the GPU and returns them as raw RGBA8 bytes.
    ///
    /// This is `async` because WebGPU buffer mapping goes through a browser
    /// Promise. On native, `device.poll(Wait)` blocks synchronously instead.
    ///
    /// The returned `Vec<u8>` is `width * height * 4` bytes, in row-major order,
    /// with R at byte 0 of each pixel (matching the `ImageData` layout expected
    /// by the canvas 2D API).
    pub async fn get_pixels(&self, width: u32, height: u32) -> Vec<u8> {
        let size = (width * height * 4) as u64;

        // Staging buffer: CPU-readable, receives a copy of the output.
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("pixel_staging"),
            size,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("readback") }
        );
        encoder.copy_buffer_to_buffer(
            self.output_buf.as_ref().expect("get_pixels called before render_frame"),
            0,
            &staging,
            0,
            size,
        );
        self.queue.submit(Some(encoder.finish()));

        // Map the staging buffer asynchronously.
        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).ok(); });

        // Drive completion: poll on native, yield to the JS event loop on WASM.
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::wait());

        #[cfg(target_arch = "wasm32")]
        {
            // A single microtask yield lets the WebGPU Promise resolve and fire
            // the map_async callback before we try to receive on rx.
            let _ = wasm_bindgen_futures::JsFuture::from(
                js_sys::Promise::resolve(&wasm_bindgen::JsValue::UNDEFINED)
            ).await;
        }

        rx.await
            .expect("map_async channel closed")
            .expect("buffer mapping failed");

        let data   = slice.get_mapped_range();
        let pixels = data.to_vec();
        drop(data);
        staging.unmap();
        pixels
    }
}
