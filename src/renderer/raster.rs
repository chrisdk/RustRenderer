//! Rasterising preview renderer.
//!
//! Path tracing is beautiful but slow — even 2-bounce preview mode is too
//! expensive for interactive orbit-camera dragging. This module provides a
//! second GPU pipeline that renders the same scene geometry using traditional
//! vertex + fragment shaders, running entirely on the GPU at 60 fps without
//! needing to trace a single ray.
//!
//! ## Architecture
//!
//! [`RasterRenderer`] owns its own GPU buffers (vertices, indices, instance
//! data, materials, depth texture, output texture) but **not** the
//! `wgpu::Device` and `wgpu::Queue` — those are borrowed from the existing
//! path-tracing [`super::Renderer`] at each call site. This avoids duplicating
//! the device/queue pair or wrapping them in `Rc`/`Arc`.
//!
//! ## Binding layout
//!
//! | Group | Binding | Name        | Type    | Shader stage |
//! |-------|---------|-------------|---------|--------------|
//! | 0     | 0       | camera      | uniform | vertex + frag |
//! | 0     | 1       | instances   | storage | vertex       |
//! | 0     | 2       | materials   | storage | fragment     |
//!
//! ## Rendering approach
//!
//! One draw call per `MeshInstance`. All vertices live in a single shared
//! vertex buffer; all indices in a single index buffer. Instance transforms
//! and material indices are packed into a `GpuInstance` storage buffer,
//! indexed by the builtin `instance_index` in the vertex shader.

use crate::camera::RasterCameraUniform;
use crate::scene::{geometry::Vertex, material::Material, Scene};

// =============================================================================
// Supporting types
// =============================================================================

/// Per-instance GPU data: the model transform + which material to shade with.
///
/// 80 bytes. Must match the `InstanceData` struct in `raster.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuInstance {
    /// Model matrix (local → world space). Column-major: `transform[col][row]`.
    transform: [[f32; 4]; 4],  // 64 bytes

    /// Index into the materials storage buffer.
    mat_index: u32,            // 4 bytes

    _pad: [u32; 3],            // 12 bytes — brings struct to 80 (a multiple of 16)
}

/// One draw call: which triangles to draw (mesh geometry) and which row of the
/// instance buffer to look up (transform + material).
#[derive(Clone, Copy)]
struct DrawCall {
    /// First index in the shared index buffer (byte offset ÷ 4).
    first_index: u32,
    /// Number of indices for this mesh (always a multiple of 3).
    index_count: u32,
    /// Index into the `GpuInstance` storage buffer.
    instance_id: u32,
}

// =============================================================================
// RasterRenderer
// =============================================================================

/// GPU state for the rasterising preview pass.
///
/// Constructed once at startup alongside the path-tracing [`super::Renderer`].
/// The `device` and `queue` are borrowed from that renderer at each call site
/// rather than stored here.
pub struct RasterRenderer {
    // ── Pipeline (built lazily on first render_frame) ────────────────────────
    bgl:      Option<wgpu::BindGroupLayout>,
    pipeline: Option<wgpu::RenderPipeline>,

    // ── Scene data (uploaded once per loaded scene) ──────────────────────────
    vertex_buf:   Option<wgpu::Buffer>,
    index_buf:    Option<wgpu::Buffer>,
    instance_buf: Option<wgpu::Buffer>,
    material_buf: Option<wgpu::Buffer>,
    draw_calls:   Vec<DrawCall>,

    // ── Per-frame data (updated before each dispatch) ────────────────────────
    camera_buf: Option<wgpu::Buffer>,

    // ── Render targets (recreated when resolution changes) ───────────────────
    output_tex: Option<wgpu::Texture>,
    depth_tex:  Option<wgpu::Texture>,
    /// Output buffer into which the texture is copied for CPU readback.
    readback_buf: Option<wgpu::Buffer>,
    output_dims: (u32, u32),
}

// =============================================================================
// Construction
// =============================================================================

impl RasterRenderer {
    /// Creates a new renderer with no scene loaded.
    ///
    /// The GPU device and queue are borrowed from the path-tracing renderer and
    /// only needed at call sites — nothing is allocated here.
    pub fn new() -> Self {
        Self {
            bgl:          None,
            pipeline:     None,
            vertex_buf:   None,
            index_buf:    None,
            instance_buf: None,
            material_buf: None,
            draw_calls:   Vec::new(),
            camera_buf:   None,
            output_tex:   None,
            depth_tex:    None,
            readback_buf: None,
            output_dims:  (0, 0),
        }
    }
}

// =============================================================================
// Scene upload
// =============================================================================

impl RasterRenderer {
    /// Uploads vertex, index, instance, and material data to the GPU.
    ///
    /// Calling this a second time drops the old buffers and replaces them.
    /// The draw call list is rebuilt from the new scene's instance table.
    pub fn upload_scene(
        &mut self,
        device:   &wgpu::Device,
        queue:    &wgpu::Queue,
        scene:    &Scene,
    ) {
        use wgpu::util::DeviceExt;

        // ── Vertex buffer ────────────────────────────────────────────────────
        self.vertex_buf = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label:    Some("raster_vertices"),
                contents: bytemuck::cast_slice(&scene.vertices),
                usage:    wgpu::BufferUsages::VERTEX,
            },
        ));

        // ── Index buffer ─────────────────────────────────────────────────────
        self.index_buf = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label:    Some("raster_indices"),
                contents: bytemuck::cast_slice(&scene.indices),
                usage:    wgpu::BufferUsages::INDEX,
            },
        ));

        // ── Instance buffer ──────────────────────────────────────────────────
        // One GpuInstance per MeshInstance, in the same order. Each entry
        // records the model transform and the material index for that instance's mesh.
        let instances: Vec<GpuInstance> = scene.instances.iter().map(|inst| {
            let mesh = &scene.meshes[inst.mesh_index as usize];
            GpuInstance {
                transform: inst.transform,
                mat_index: mesh.material_index,
                _pad: [0; 3],
            }
        }).collect();

        self.instance_buf = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label:    Some("raster_instances"),
                contents: bytemuck::cast_slice(&instances),
                usage:    wgpu::BufferUsages::STORAGE,
            },
        ));

        // ── Material buffer ──────────────────────────────────────────────────
        // WebGPU doesn't allow zero-size storage buffers; if the scene has no
        // materials (unusual but legal GLTF), supply a dummy default material.
        // The dummy must outlive the buffer_init call below, so it's a local.
        let dummy_material;
        let mat_data: &[Material] = if scene.materials.is_empty() {
            dummy_material = Material::default();
            std::slice::from_ref(&dummy_material)
        } else {
            &scene.materials
        };
        self.material_buf = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label:    Some("raster_materials"),
                contents: bytemuck::cast_slice(mat_data),
                usage:    wgpu::BufferUsages::STORAGE,
            },
        ));

        // ── Draw call list ───────────────────────────────────────────────────
        // Each MeshInstance becomes one draw call referencing its mesh's
        // index range and its own row in the instance buffer.
        self.draw_calls = scene.instances.iter().enumerate().map(|(i, inst)| {
            let mesh = &scene.meshes[inst.mesh_index as usize];
            DrawCall {
                first_index: mesh.first_index,
                index_count: mesh.index_count,
                instance_id: i as u32,
            }
        }).collect();

        // Reset render targets so they're recreated with the correct pipeline
        // on the next render_frame call.
        self.output_dims  = (0, 0);
        self.output_tex   = None;
        self.depth_tex    = None;
        self.readback_buf = None;

        log::info!(
            "raster scene uploaded: {} vertices, {} instances → {} draw calls",
            scene.vertices.len(), scene.instances.len(), self.draw_calls.len(),
        );

        // ── Lazily initialise / rebuild pipeline ─────────────────────────────
        // We must rebuild here because the old bind group layout (if any) is
        // still valid, but some callers may call upload_scene before ever
        // calling render_frame, so we build eagerly to surface shader errors
        // during loading rather than during the first frame.
        let _ = queue;  // queue not needed for buffer uploads (create_buffer_init handles it)
        self.build_pipeline(device);
    }

    /// Returns `true` once [`upload_scene`] has been called.
    pub fn is_scene_loaded(&self) -> bool {
        self.vertex_buf.is_some()
    }
}

// =============================================================================
// Camera upload
// =============================================================================

impl RasterRenderer {
    /// Uploads (or updates) the raster camera uniform buffer.
    pub fn upload_camera(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, uniform: &RasterCameraUniform) {
        let data = bytemuck::bytes_of(uniform);
        match &self.camera_buf {
            Some(buf) => queue.write_buffer(buf, 0, data),
            None => {
                use wgpu::util::DeviceExt;
                self.camera_buf = Some(device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label:    Some("raster_camera"),
                        contents: data,
                        usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    },
                ));
            }
        }
    }
}

// =============================================================================
// Pipeline construction
// =============================================================================

impl RasterRenderer {
    fn build_pipeline(&mut self, device: &wgpu::Device) {
        // ── Shader module ────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("raster.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/raster.wgsl").into()
            ),
        });

        // ── Bind group layout ────────────────────────────────────────────────
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("raster_bgl"),
            entries: &[
                // Binding 0: camera uniform (vertex + fragment)
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // Binding 1: instance transforms (vertex only)
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // Binding 2: materials (fragment only)
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        // ── Pipeline layout ──────────────────────────────────────────────────
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:               Some("raster_layout"),
            bind_group_layouts:  &[&bgl],
            immediate_size:      0,
        });

        // ── Vertex buffer layout (must match Vertex struct: 48 bytes) ────────
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { shader_location: 0, offset:  0, format: wgpu::VertexFormat::Float32x3 }, // position
                wgpu::VertexAttribute { shader_location: 1, offset: 12, format: wgpu::VertexFormat::Float32x3 }, // normal
                wgpu::VertexAttribute { shader_location: 2, offset: 24, format: wgpu::VertexFormat::Float32x2 }, // uv
                wgpu::VertexAttribute { shader_location: 3, offset: 32, format: wgpu::VertexFormat::Float32x4 }, // tangent
            ],
        };

        // ── Render pipeline ──────────────────────────────────────────────────
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("raster_pipeline"),
            layout: Some(&layout),

            vertex: wgpu::VertexState {
                module:             &shader,
                entry_point:        Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers:            &[vertex_layout],
            },

            fragment: Some(wgpu::FragmentState {
                module:             &shader,
                entry_point:        Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    // Rgba8Unorm to match the path tracer output format, so both
                    // pipelines feed pixels through the same JS-side get_pixels path.
                    format:     wgpu::TextureFormat::Rgba8Unorm,
                    blend:      None,  // opaque; alpha blending would need sorted draw calls
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),

            primitive: wgpu::PrimitiveState {
                topology:          wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face:        wgpu::FrontFace::Ccw,  // GLTF convention
                // Cull back faces. Glass shells look correct from outside; if a
                // glass model appears hollow, the path tracer will show the inside.
                cull_mode:         Some(wgpu::Face::Back),
                unclipped_depth:   false,
                polygon_mode:      wgpu::PolygonMode::Fill,
                conservative:      false,
            },

            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare:       wgpu::CompareFunction::Less,
                stencil:             wgpu::StencilState::default(),
                bias:                wgpu::DepthBiasState::default(),
            }),

            multisample:   wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        self.bgl      = Some(bgl);
        self.pipeline = Some(pipeline);

        log::info!("raster pipeline ready");
    }
}

// =============================================================================
// Rendering
// =============================================================================

impl RasterRenderer {
    /// Draws all scene instances and stores the result in an internal
    /// output texture.
    ///
    /// Panics if [`upload_scene`] or [`upload_camera`] has not been called.
    pub fn render_frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) {
        assert!(self.is_scene_loaded(), "render_frame called before upload_scene");
        assert!(self.camera_buf.is_some(), "render_frame called before upload_camera");
        assert!(self.pipeline.is_some(), "render_frame called before pipeline was built");

        // ── (Re)create render targets when resolution changes ────────────────
        if self.output_dims != (width, height) {
            // How many bytes per row, rounded up to the WebGPU copy alignment.
            // We store this so get_pixels knows how to strip padding.
            self.output_tex = Some(device.create_texture(&wgpu::TextureDescriptor {
                label:           Some("raster_output"),
                size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count:    1,
                dimension:       wgpu::TextureDimension::D2,
                format:          wgpu::TextureFormat::Rgba8Unorm,
                usage:           wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats:    &[],
            }));

            self.depth_tex = Some(device.create_texture(&wgpu::TextureDescriptor {
                label:           Some("raster_depth"),
                size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count:    1,
                dimension:       wgpu::TextureDimension::D2,
                format:          wgpu::TextureFormat::Depth32Float,
                usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats:    &[],
            }));

            // Readback buffer: must use the aligned bytes-per-row.
            let aligned_row = aligned_bytes_per_row(width);
            self.readback_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("raster_readback"),
                size:               (aligned_row * height) as u64,
                usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));

            self.output_dims = (width, height);
        }

        let output_view = self.output_tex.as_ref().unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = self.depth_tex.as_ref().unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());

        // ── Build bind group ─────────────────────────────────────────────────
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("raster_bg"),
            layout:  self.bgl.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: self.camera_buf.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: self.instance_buf.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: self.material_buf.as_ref().unwrap().as_entire_binding(),
                },
            ],
        });

        // ── Encode render pass ───────────────────────────────────────────────
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("raster") }
        );

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raster_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &output_view,
                    resolve_target: None,
                    depth_slice:    None,  // None for 2D textures (only used for 3D texture slices)
                    ops: wgpu::Operations {
                        // Sky blue-grey clear colour — consistent with the path
                        // tracer's sky at the horizon so the background matches.
                        load:  wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.18, g: 0.20, b: 0.25, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view:         &depth_view,
                    depth_ops:    Some(wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,  // depth not needed after pass
                    }),
                    stencil_ops:  None,
                }),
                timestamp_writes:    None,
                occlusion_query_set: None,
                multiview_mask:      None,
            });

            pass.set_pipeline(self.pipeline.as_ref().unwrap());
            pass.set_bind_group(0, &bg, &[]);
            pass.set_vertex_buffer(0, self.vertex_buf.as_ref().unwrap().slice(..));
            pass.set_index_buffer(
                self.index_buf.as_ref().unwrap().slice(..),
                wgpu::IndexFormat::Uint32,
            );

            // One draw call per instance. We use the *logical* instance index
            // (not a range of GPU instances) so the vertex shader can look up
            // the correct row in the instance storage buffer.
            for dc in &self.draw_calls {
                pass.draw_indexed(
                    dc.first_index .. dc.first_index + dc.index_count,
                    0,  // base_vertex: indices already point into the flat vertex buffer
                    dc.instance_id .. dc.instance_id + 1,
                );
            }
        }

        queue.submit(Some(encoder.finish()));
        log::debug!("raster_frame dispatched ({}×{})", width, height);
    }

    /// Copies the rendered pixels from the GPU and returns them as raw RGBA8 bytes.
    ///
    /// Must be called after [`render_frame`]. Returns `width * height * 4` bytes
    /// in row-major, RGBA order — the same format the JS `ImageData` API expects.
    pub async fn get_pixels(&self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> Vec<u8> {
        let aligned_row = aligned_bytes_per_row(width);
        let tight_row   = width * 4;

        let readback = self.readback_buf.as_ref()
            .expect("get_pixels called before render_frame");
        let output_tex = self.output_tex.as_ref()
            .expect("get_pixels called before render_frame");

        // Copy texture → readback buffer (with padding to meet alignment).
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("raster_readback") }
        );
        // wgpu 28 renamed ImageCopyBuffer → TexelCopyBufferInfo and
        // ImageDataLayout → TexelCopyBufferLayout.
        encoder.copy_texture_to_buffer(
            output_tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset:         0,
                    bytes_per_row:  Some(aligned_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));

        // Map the readback buffer.
        let slice = readback.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });

        #[cfg(not(target_arch = "wasm32"))]
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        #[cfg(target_arch = "wasm32")]
        {
            let _ = wasm_bindgen_futures::JsFuture::from(
                js_sys::Promise::resolve(&wasm_bindgen::JsValue::UNDEFINED)
            ).await;
        }

        rx.await
            .expect("raster map_async channel closed")
            .expect("raster buffer mapping failed");

        let raw = slice.get_mapped_range();

        // Strip the row padding: the GPU wrote `aligned_row` bytes per row but
        // we want `tight_row` bytes per row (exactly width * 4).
        let mut out = Vec::with_capacity((tight_row * height) as usize);
        for row in 0..height {
            let start = (row * aligned_row) as usize;
            let end   = start + tight_row as usize;
            out.extend_from_slice(&raw[start..end]);
        }

        drop(raw);
        readback.unmap();
        out
    }
}

// =============================================================================
// Private helpers
// =============================================================================

/// WebGPU requires that `bytes_per_row` in `copy_texture_to_buffer` is a
/// multiple of `COPY_BYTES_PER_ROW_ALIGNMENT` (256). Rounds up to that.
fn aligned_bytes_per_row(width: u32) -> u32 {
    const ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let tight = width * 4;  // 4 bytes per RGBA8 pixel
    (tight + ALIGN - 1) & !(ALIGN - 1)
}
