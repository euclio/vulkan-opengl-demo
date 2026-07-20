use std::{iter, ptr};

use ash::vk;
use gl::types::GLuint;
use wgpu::hal::api::Vulkan;
use wgpu::rwh::RawDisplayHandle;
use wgpu::{
    BackendOptions, CurrentSurfaceTexture, InstanceFlags, MemoryBudgetThresholds, TextureUses, hal,
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::{Window, WindowAttributes};

pub static EGL: egl::Instance<egl::Static> = egl::Instance::new(egl::Static);

/// Checks for GL errors after executing the given expression.
macro_rules! gl_call {
    ($e:expr) => {
        if cfg!(debug_assertions) {
            // Clear any existing errors.
            while gl::GetError() != gl::NO_ERROR {}

            let result = $e;

            let mut errored = false;

            loop {
                let err = gl::GetError();

                if err == gl::NO_ERROR {
                    break;
                }

                eprintln!("OpenGL error: {:#X} at {}:{}", err, file!(), line!());

                errored = true;
            }

            if errored {
                panic!("Encountered OpenGL error in {}", stringify!($e));
            }

            result
        } else {
            $e
        }
    };
}

mod gl {
    #![allow(unsafe_op_in_unsafe_fn)]
    #![allow(clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));
}

/// Texture memory shared between Vulkan and OpenGL.
#[derive(Debug)]
struct SharedTexture {
    /// wgpu Texture.
    texture: wgpu::Texture,

    /// GL handle to Vulkan texture memory.
    gl_mem_object: GLuint,

    /// GL texture ID.
    gl_texture: GLuint,
}

/// Semaphores shared between Vulkan and OpenGL for synchronization.
struct Semaphores {
    /// GL is finished writing.
    gl_write_finished: ash::vk::Semaphore,

    // GL-imported [`gl_write_finished`].
    gl_gl_write_finished: u32,

    // VK is finished reading.
    vk_read_finished: ash::vk::Semaphore,

    // GL-imported [`vk_read_finished`].
    gl_vk_read_finished: u32,
}

struct State {
    configured: bool,

    width: u32,
    height: u32,

    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    /// Surface format.
    format: wgpu::TextureFormat,
    render_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,

    shared_texture: Option<SharedTexture>,
    semaphores: Semaphores,

    window: Window, // Must be dropped last
}

impl State {
    async fn new(window: Window) -> Self {
        let hal_instance = unsafe {
            hal::vulkan::Instance::init_with_callback(
                &hal::InstanceDescriptor {
                    name: "wgpu",
                    flags: InstanceFlags::VALIDATION,
                    memory_budget_thresholds: MemoryBudgetThresholds::default(),
                    backend_options: BackendOptions::from_env_or_default(),
                    telemetry: None,
                    display: None,
                },
                Some(Box::new(|args| {
                    args.extensions.extend_from_slice(&[
                        vk::KHR_EXTERNAL_MEMORY_CAPABILITIES_NAME,
                        vk::KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_NAME,
                    ])
                })),
            )
            .unwrap()
        };
        let instance = unsafe { wgpu::Instance::from_hal::<Vulkan>(hal_instance) };

        let hal_instance = unsafe { instance.as_hal::<Vulkan>() }.unwrap();
        let raw_vk_instance = hal_instance.shared_instance().raw_instance();

        let raw_display_handle = window.display_handle().unwrap().as_raw();

        let raw_window_handle = window.window_handle().unwrap().as_raw();

        let surface = unsafe {
            instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle: Some(raw_display_handle),
                    raw_window_handle,
                })
                .unwrap()
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("Failed to find a suitable adapter");

        let surface_capabilities = surface.get_capabilities(&adapter);
        let format = surface_capabilities.formats[0];

        let hal_adapter = unsafe { adapter.as_hal::<Vulkan>().unwrap() };
        let _raw_physical_device = hal_adapter.raw_physical_device();

        let open_device = unsafe {
            hal_adapter.open_with_callback(
                wgpu::Features::default(),
                &wgpu::Limits::default(),
                &wgpu::MemoryHints::default(),
                Some(Box::new(|args| {
                    args.extensions
                        .extend_from_slice(&[vk::KHR_EXTERNAL_SEMAPHORE_FD_NAME]);
                })),
            )
        }
        .unwrap();

        let (device, queue) = unsafe {
            adapter
                .create_device_from_hal::<Vulkan>(open_device, &wgpu::DeviceDescriptor::default())
        }
        .expect("Failed to request device");

        let hal_device = unsafe { device.as_hal::<Vulkan>().unwrap() };
        let raw_vk_device = hal_device.raw_device();

        let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dummy texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            dimension: wgpu::TextureDimension::D2,
            format,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });

        let dummy_texture_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dummy_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shader.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let display_ptr = match raw_display_handle {
            RawDisplayHandle::Wayland(wayland) => wayland.display.as_ptr(),
            RawDisplayHandle::Xlib(xlib) => xlib.display.as_ref().unwrap().as_ptr(),
            _ => unimplemented!(),
        };

        let egl_display = unsafe { EGL.get_display(display_ptr) }.unwrap();
        EGL.initialize(egl_display).unwrap();
        EGL.bind_api(egl::OPENGL_API).unwrap();

        let egl_config = EGL
            .choose_first_config(
                egl_display,
                &[
                    egl::SURFACE_TYPE,
                    egl::PBUFFER_BIT,
                    egl::RED_SIZE,
                    8,
                    egl::GREEN_SIZE,
                    8,
                    egl::BLUE_SIZE,
                    8,
                    egl::DEPTH_SIZE,
                    24,
                    egl::RENDERABLE_TYPE,
                    egl::OPENGL_BIT,
                    egl::NONE,
                ],
            )
            .unwrap()
            .unwrap();

        let ctx = EGL
            .create_context(
                egl_display,
                egl_config,
                None,
                &[
                    egl::CONTEXT_MAJOR_VERSION,
                    4,
                    egl::CONTEXT_MINOR_VERSION,
                    6,
                    egl::CONTEXT_OPENGL_PROFILE_MASK,
                    egl::CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
                    egl::NONE,
                ],
            )
            .unwrap();

        EGL.make_current(egl_display, None, None, Some(ctx))
            .unwrap();

        gl::load_with(|symbol| EGL.get_proc_address(symbol).unwrap() as *const _);
        let mut export_sem_info = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut export_sem_info);
        let gl_write_finished = unsafe { raw_vk_device.create_semaphore(&sem_info, None) }.unwrap();
        let vk_read_finished = unsafe { raw_vk_device.create_semaphore(&sem_info, None) }.unwrap();

        let ext_sem_fn =
            ash::khr::external_semaphore_fd::Device::new(raw_vk_instance, raw_vk_device);
        let gl_write_finished_fd = unsafe {
            ext_sem_fn
                .get_semaphore_fd(
                    &vk::SemaphoreGetFdInfoKHR::default()
                        .semaphore(gl_write_finished)
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
                )
                .unwrap()
        };
        let vk_read_finished_fd = unsafe {
            ext_sem_fn
                .get_semaphore_fd(
                    &vk::SemaphoreGetFdInfoKHR::default()
                        .semaphore(vk_read_finished)
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
                )
                .unwrap()
        };

        // Import the semaphores into GL.
        let mut gl_gl_write_finished = 0;
        let mut gl_vk_read_finished = 0;
        unsafe {
            gl_call!(gl::GenSemaphoresEXT(1, &mut gl_gl_write_finished));
            gl_call!(gl::ImportSemaphoreFdEXT(
                gl_gl_write_finished,
                gl::HANDLE_TYPE_OPAQUE_FD_EXT,
                gl_write_finished_fd,
            ));
            gl_call!(gl::GenSemaphoresEXT(1, &mut gl_vk_read_finished));
            gl_call!(gl::ImportSemaphoreFdEXT(
                gl_vk_read_finished,
                gl::HANDLE_TYPE_OPAQUE_FD_EXT,
                vk_read_finished_fd,
            ));
        }

        let hal_queue = unsafe { queue.as_hal::<Vulkan>() }.unwrap();

        // Pre-signal so GL can proceed on the first frame.
        hal_queue.add_signal_semaphore(vk_read_finished, None);
        queue.submit(iter::empty());

        let semaphores = Semaphores {
            gl_write_finished,
            gl_gl_write_finished,
            vk_read_finished,
            gl_vk_read_finished,
        };

        State {
            configured: false,

            width: 256,
            height: 256,
            window,
            instance,
            device,
            surface,
            format,
            adapter,
            queue,
            render_pipeline,
            sampler,
            bind_group_layout,
            bind_group,

            shared_texture: None,
            semaphores,
        }
    }

    fn render(&self) {
        let shared_texture = self.shared_texture.as_ref().unwrap();
        let gl_texture_id = shared_texture.gl_texture;

        unsafe {
            gl_call!(gl::WaitSemaphoreEXT(
                self.semaphores.gl_vk_read_finished,
                0,
                ptr::null(),
                1,
                &gl_texture_id,
                &gl::LAYOUT_GENERAL_EXT,
            ));

            gl_call!(gl::ClearColor(0.0, 0.0, 0.0, 1.0));
            gl_call!(gl::Clear(gl::COLOR_BUFFER_BIT));

            gl::Begin(gl::TRIANGLES);
            {
                gl::Color3f(1.0, 0.0, 0.0);
                gl::Vertex2f(0.0, 1.0);

                gl::Color3f(0.0, 1.0, 0.0);
                gl::Vertex2f(-1.0, -1.0);

                gl::Color3f(0.0, 0.0, 1.0);
                gl::Vertex2f(1.0, -1.0);
            }
            gl::End();

            gl_call!(gl::Flush());

            // Signal Vulkan that GL is finished
            gl_call!(gl::SignalSemaphoreEXT(
                self.semaphores.gl_gl_write_finished,
                0,
                ptr::null(),
                1,
                &gl_texture_id,
                &gl::LAYOUT_COLOR_ATTACHMENT_EXT,
            ));
        }

        let device = &self.device;

        let surface = &self.surface;

        let frame = match surface.get_current_texture() {
            CurrentSurfaceTexture::Success(frame) => frame,
            other => panic!("unable to get frame: {:?}", other),
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let queue = &self.queue;
        let hal_queue = unsafe { queue.as_hal::<Vulkan>() }.unwrap();

        hal_queue.add_wait_semaphore(
            self.semaphores.gl_write_finished,
            None,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        );

        let mut encoder =
            device.create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor::default());

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("OpenGL pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }

        let command_buffer = encoder.finish();

        hal_queue.add_signal_semaphore(self.semaphores.vk_read_finished, None);

        queue.submit(iter::once(command_buffer));

        queue.present(frame);
    }

    fn resize(&mut self, new_size: (Option<u32>, Option<u32>)) {
        let (new_width, new_height) = new_size;
        self.width = new_width.unwrap_or(256);
        self.height = new_height.unwrap_or(256);

        let instance = &self.instance;
        let adapter = &self.adapter;
        let surface = &self.surface;
        let device = &self.device;

        if self.shared_texture.is_none() || matches!(new_size, (Some(_), Some(_))) {
            if let Some(old) = self.shared_texture.take() {
                unsafe {
                    gl_call!(gl::DeleteTextures(1, &old.gl_texture));
                    gl_call!(gl::DeleteMemoryObjectsEXT(1, &old.gl_mem_object));
                }

                old.texture.destroy();
            }

            let hal_instance = unsafe { instance.as_hal::<Vulkan>() }.unwrap();
            let raw_instance = hal_instance.shared_instance().raw_instance();

            let hal_adapter = unsafe { adapter.as_hal::<Vulkan>() }.unwrap();
            let vk_physical_device = hal_adapter.raw_physical_device();

            let hal_device = unsafe { device.as_hal::<Vulkan>() }.unwrap();
            let raw_device = hal_device.raw_device();

            // OpenGL expects RGBA, so allocate the texture with a matching format.
            let (format, vk_format, gl_format) = if self.format.is_srgb() {
                (
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    vk::Format::R8G8B8A8_SRGB,
                    gl::SRGB8_ALPHA8,
                )
            } else {
                (
                    wgpu::TextureFormat::Rgba8Unorm,
                    vk::Format::R8G8B8A8_UNORM,
                    gl::RGBA8,
                )
            };

            let vk_image = unsafe {
                raw_device.create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk_format)
                        .extent(vk::Extent3D {
                            width: self.width,
                            height: self.height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
                        .push_next(
                            &mut vk::ExternalMemoryImageCreateInfo::default()
                                .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD),
                        ),
                    None,
                )
            }
            .unwrap();

            let mem_reqs = unsafe { raw_device.get_image_memory_requirements(vk_image) };
            let mem_properties =
                unsafe { raw_instance.get_physical_device_memory_properties(vk_physical_device) };

            let memory_type_index = (0..mem_properties.memory_type_count)
                .into_iter()
                .find(|i| {
                    let is_type_supported = (mem_reqs.memory_type_bits & (1 << i)) != 0;

                    let has_required_properties = mem_properties.memory_types[*i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);

                    is_type_supported && has_required_properties
                })
                .unwrap();

            let vk_memory = unsafe {
                raw_device.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(mem_reqs.size)
                        .memory_type_index(memory_type_index)
                        .push_next(
                            &mut vk::ExportMemoryAllocateInfo::default()
                                .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD),
                        )
                        .push_next(&mut vk::MemoryDedicatedAllocateInfo::default().image(vk_image)),
                    None,
                )
            }
            .unwrap();

            unsafe { raw_device.bind_image_memory(vk_image, vk_memory, 0) }.unwrap();

            let external_memory_fd =
                ash::khr::external_memory_fd::Device::new(raw_instance, raw_device);
            let fd_info = ash::vk::MemoryGetFdInfoKHR::default()
                .memory(vk_memory)
                .handle_type(ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
            let fd = unsafe { external_memory_fd.get_memory_fd(&fd_info) }.unwrap();

            let mut gl_texture = 0;
            let mut gl_mem_object = 0;
            unsafe {
                gl_call!(gl::CreateMemoryObjectsEXT(1, &mut gl_mem_object));

                let dedicated = gl::TRUE.into();
                gl_call!(gl::MemoryObjectParameterivEXT(
                    gl_mem_object,
                    gl::DEDICATED_MEMORY_OBJECT_EXT,
                    &dedicated
                ));
                gl_call!(gl::ImportMemoryFdEXT(
                    gl_mem_object,
                    mem_reqs.size,
                    gl::HANDLE_TYPE_OPAQUE_FD_EXT,
                    fd
                ));

                gl_call!(gl::GenTextures(1, &mut gl_texture));
                gl_call!(gl::BindTexture(gl::TEXTURE_2D, gl_texture));
                gl_call!(gl::TexParameteri(
                    gl::TEXTURE_2D,
                    gl::TEXTURE_TILING_EXT,
                    gl::OPTIMAL_TILING_EXT.try_into().unwrap()
                ));
                gl_call!(gl::TextureStorageMem2DEXT(
                    gl_texture,
                    1,
                    gl_format,
                    self.width.try_into().unwrap(),
                    self.height.try_into().unwrap(),
                    gl_mem_object,
                    0,
                ));

                let mut fbo = 0;
                gl_call!(gl::GenFramebuffers(1, &mut fbo));
                gl_call!(gl::BindFramebuffer(gl::FRAMEBUFFER, fbo));
                gl_call!(gl::FramebufferTexture2D(
                    gl::FRAMEBUFFER,
                    gl::COLOR_ATTACHMENT0,
                    gl::TEXTURE_2D,
                    gl_texture,
                    0,
                ));

                gl_call!(gl::Viewport(
                    0,
                    0,
                    self.width.try_into().unwrap(),
                    self.height.try_into().unwrap()
                ));
            }

            let hal_texture = unsafe {
                hal_device.texture_from_raw(
                    vk_image,
                    &hal::TextureDescriptor {
                        label: Some("Shared Hal texture"),
                        size: wgpu::Extent3d {
                            width: self.width,
                            height: self.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: TextureUses::COLOR_TARGET | TextureUses::RESOURCE,
                        memory_flags: hal::MemoryFlags::empty(),
                        view_formats: vec![],
                    },
                    None,
                    hal::vulkan::TextureMemory::Dedicated(vk_memory),
                )
            };

            let texture = unsafe {
                device.create_texture_from_hal::<Vulkan>(
                    hal_texture,
                    &wgpu::TextureDescriptor {
                        label: Some("OpenGL target texture"),
                        size: wgpu::Extent3d {
                            width: self.width,
                            height: self.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                    wgpu::TextureUses::UNINITIALIZED,
                )
            };

            let shared_texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

            self.shared_texture = Some(SharedTexture {
                texture,
                gl_mem_object,
                gl_texture,
            });

            self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("OpenGL Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&shared_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
        }

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.format,
            view_formats: vec![self.format],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: self.width,
            height: self.height,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::Fifo,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };

        surface.configure(&self.device, &surface_config);

        self.configured = true;
    }
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.state.is_none() {
            let attrs = WindowAttributes::default().with_title("Vulkan OpenGL Demo");
            let window = event_loop.create_window(attrs).unwrap();

            let state = pollster::block_on(State::new(window));
            self.state = Some(state);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                state.resize((Some(physical_size.width), Some(physical_size.height)));
                state.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                state.render();
                state.window.request_redraw();
            }
            _ => (),
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
