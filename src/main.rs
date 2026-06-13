use std::ffi::c_void;
use std::ptr::NonNull;
use std::time::Duration;
use std::{iter, ptr};

use ash::vk;
use gl::types::GLuint;
use smithay_client_toolkit::compositor::{CompositorHandler, CompositorState};
use smithay_client_toolkit::output::{OutputHandler, OutputState};
use smithay_client_toolkit::reexports::calloop::{EventLoop, LoopHandle};
use smithay_client_toolkit::reexports::calloop_wayland_source::WaylandSource;
use smithay_client_toolkit::reexports::client::globals::registry_queue_init;
use smithay_client_toolkit::reexports::client::{Connection, Proxy};
use smithay_client_toolkit::registry::{ProvidesRegistryState, RegistryState};
use smithay_client_toolkit::seat::{SeatHandler, SeatState};
use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::shell::xdg::XdgShell;
use smithay_client_toolkit::shell::xdg::window::{Window, WindowHandler};
use smithay_client_toolkit::{
    delegate_compositor, delegate_output, delegate_registry, delegate_seat, delegate_xdg_shell,
    delegate_xdg_window, registry_handlers,
};
use wgpu::hal::api::Vulkan;
use wgpu::rwh::{RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle};
use wgpu::{
    BackendOptions, CurrentSurfaceTexture, InstanceFlags, MemoryBudgetThresholds, PollType,
    TextureUses,
};
use wgpu_hal::vulkan::{Instance, TextureMemory};

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
    /// Backing EGL image.
    egl_image: egl::Image,

    /// wgpu Texture.
    texture: wgpu::Texture,

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
    loop_handle: LoopHandle<'static, Self>,
    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
    configured: bool,

    exit: bool,
    width: u32,
    height: u32,
    window: Window,

    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    render_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,

    egl_display: egl::Display,
    shared_texture: Option<SharedTexture>,
    semaphores: Semaphores,
}

impl State {
    fn queue_render(&self) {
        if !self.configured {
            return;
        }

        self.loop_handle.insert_idle(|state| {
            state.render();
        });
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
        let hal_device = unsafe { device.as_hal::<Vulkan>() }.unwrap();
        let raw_device = hal_device.raw_device();

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
        let raw_queue = hal_queue.as_raw();

        unsafe {
            raw_device.queue_submit(
                raw_queue,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&[self.semaphores.gl_write_finished])
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::FRAGMENT_SHADER])],
                vk::Fence::null(),
            )
        }
        .unwrap();

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

        let submission = queue.submit(iter::once(command_buffer));

        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .unwrap();

        unsafe {
            raw_device
                .queue_submit(
                    raw_queue,
                    &[vk::SubmitInfo::default()
                        .signal_semaphores(&[self.semaphores.vk_read_finished])],
                    vk::Fence::null(),
                )
                .unwrap();
        }

        frame.present();
        self.queue_render();
    }
}

impl CompositorHandler for State {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &wayland_client::QueueHandle<Self>,
        _surface: &wayland_client::protocol::wl_surface::WlSurface,
        _new_factor: i32,
    ) {
    }

    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &wayland_client::QueueHandle<Self>,
        _surface: &wayland_client::protocol::wl_surface::WlSurface,
        _new_transform: wayland_client::protocol::wl_output::Transform,
    ) {
    }

    fn frame(
        &mut self,
        _conn: &Connection,
        _qh: &wayland_client::QueueHandle<Self>,
        _surface: &wayland_client::protocol::wl_surface::WlSurface,
        _time: u32,
    ) {
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &wayland_client::QueueHandle<Self>,
        _surface: &wayland_client::protocol::wl_surface::WlSurface,
        _output: &wayland_client::protocol::wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &wayland_client::QueueHandle<Self>,
        _surface: &wayland_client::protocol::wl_surface::WlSurface,
        _output: &wayland_client::protocol::wl_output::WlOutput,
    ) {
    }
}

impl OutputHandler for State {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(
        &mut self,
        _conn: &Connection,
        _qh: &wayland_client::QueueHandle<Self>,
        _output: wayland_client::protocol::wl_output::WlOutput,
    ) {
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &wayland_client::QueueHandle<Self>,
        _output: wayland_client::protocol::wl_output::WlOutput,
    ) {
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &wayland_client::QueueHandle<Self>,
        _output: wayland_client::protocol::wl_output::WlOutput,
    ) {
    }
}

impl WindowHandler for State {
    fn request_close(
        &mut self,
        _conn: &Connection,
        _qh: &wayland_client::QueueHandle<Self>,
        _window: &Window,
    ) {
        self.exit = true;
    }

    fn configure(
        &mut self,
        _conn: &Connection,
        _qh: &wayland_client::QueueHandle<Self>,
        _window: &Window,
        configure: smithay_client_toolkit::shell::xdg::window::WindowConfigure,
        _serial: u32,
    ) {
        let is_initial_configure = !self.configured;

        let (new_width, new_height) = configure.new_size;
        self.width = new_width.map_or(256, |v| v.get());
        self.height = new_height.map_or(256, |v| v.get());

        let instance = &self.instance;
        let adapter = &self.adapter;
        let surface = &self.surface;
        let device = &self.device;

        if self.shared_texture.is_none() || matches!(configure.new_size, (Some(_), Some(_))) {
            // Wait for rendering to finish.
            device.poll(PollType::wait_indefinitely()).unwrap();

            if let Some(old) = self.shared_texture.take() {
                unsafe {
                    gl_call!(gl::DeleteTextures(1, &old.gl_texture));
                    EGL.destroy_image(self.egl_display, old.egl_image).unwrap();
                }

                old.texture.destroy();
            }

            let hal_instance = unsafe { instance.as_hal::<Vulkan>() }.unwrap();
            let raw_instance = hal_instance.shared_instance().raw_instance();

            let hal_adapter = unsafe { adapter.as_hal::<Vulkan>() }.unwrap();
            let vk_physical_device = hal_adapter.raw_physical_device();

            let hal_device = unsafe { device.as_hal::<Vulkan>() }.unwrap();
            let raw_device = hal_device.raw_device();

            let vk_image = unsafe {
                raw_device.create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::B8G8R8A8_UNORM)
                        .extent(vk::Extent3D {
                            width: self.width,
                            height: self.height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::LINEAR)
                        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .push_next(
                            &mut vk::ExternalMemoryImageCreateInfo::default()
                                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT),
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
                                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT),
                        )
                        .push_next(&mut vk::MemoryDedicatedAllocateInfo::default().image(vk_image)),
                    None,
                )
            }
            .unwrap();

            unsafe { raw_device.bind_image_memory(vk_image, vk_memory, 0) }.unwrap();

            let subresource_layout = unsafe {
                raw_device.get_image_subresource_layout(
                    vk_image,
                    vk::ImageSubresource::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .array_layer(0),
                )
            };

            let external_memory_fd =
                ash::khr::external_memory_fd::Device::new(raw_instance, raw_device);
            let fd_info = ash::vk::MemoryGetFdInfoKHR::default()
                .memory(vk_memory)
                .handle_type(ash::vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let fd = unsafe { external_memory_fd.get_memory_fd(&fd_info) }.unwrap();

            let egl_image = EGL
                .create_image(
                    self.egl_display,
                    unsafe { egl::Context::from_ptr(ptr::null_mut()) },
                    egl::LINUX_DMA_BUF_EXT as u32,
                    unsafe { egl::ClientBuffer::from_ptr(ptr::null_mut()) },
                    &[
                        egl::WIDTH as egl::Attrib,
                        self.width as egl::Attrib,
                        egl::HEIGHT as egl::Attrib,
                        self.height as egl::Attrib,
                        egl::LINUX_DRM_FOURCC_EXT as egl::Attrib,
                        0x34325241, // DRM_FORMAT_ARGB8888 = AR24
                        egl::DMA_BUF_PLANE0_FD_EXT as egl::Attrib,
                        fd as egl::Attrib,
                        egl::DMA_BUF_PLANE0_OFFSET_EXT as egl::Attrib,
                        subresource_layout.offset as egl::Attrib,
                        egl::DMA_BUF_PLANE0_PITCH_EXT as egl::Attrib,
                        subresource_layout.row_pitch as egl::Attrib,
                        egl::ATTRIB_NONE,
                    ],
                )
                .unwrap();

            let mut gl_texture = 0;
            unsafe {
                gl_call!(gl::GenTextures(1, &mut gl_texture));
                gl_call!(gl::BindTexture(gl::TEXTURE_2D, gl_texture));
                let gl_egl_image_target_texture: unsafe extern "C" fn(u32, *const c_void) =
                    std::mem::transmute(
                        EGL.get_proc_address("glEGLImageTargetTexture2DOES")
                            .unwrap(),
                    );
                gl_call!(gl_egl_image_target_texture(
                    gl::TEXTURE_2D,
                    egl_image.as_ptr()
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

                gl_call!(gl::Viewport(0, 0, self.width as i32, self.height as i32,));
            }

            let hal_texture = unsafe {
                hal_device.texture_from_raw(
                    vk_image,
                    &wgpu_hal::TextureDescriptor {
                        label: Some("Shared Hal texture"),
                        size: wgpu::Extent3d {
                            width: self.width,
                            height: self.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Bgra8Unorm,
                        usage: TextureUses::COLOR_TARGET | TextureUses::RESOURCE,
                        memory_flags: wgpu_hal::MemoryFlags::empty(),
                        view_formats: vec![],
                    },
                    None,
                    TextureMemory::Dedicated(vk_memory),
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
                        format: wgpu::TextureFormat::Bgra8Unorm,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                )
            };

            let shared_texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

            self.shared_texture = Some(SharedTexture {
                texture,
                gl_texture,
                egl_image,
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

        let cap = surface.get_capabilities(adapter);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: cap.formats[0],
            view_formats: vec![cap.formats[0]],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: self.width,
            height: self.height,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::Fifo,
        };

        surface.configure(&self.device, &surface_config);

        self.configured = true;

        if is_initial_configure {
            self.queue_render();
        }
    }
}

impl SeatHandler for State {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }

    fn new_seat(
        &mut self,
        _conn: &Connection,
        _qh: &wayland_client::QueueHandle<Self>,
        _seat: wayland_client::protocol::wl_seat::WlSeat,
    ) {
    }

    fn new_capability(
        &mut self,
        _conn: &Connection,
        _qh: &wayland_client::QueueHandle<Self>,
        _seat: wayland_client::protocol::wl_seat::WlSeat,
        _capability: smithay_client_toolkit::seat::Capability,
    ) {
    }

    fn remove_capability(
        &mut self,
        _conn: &Connection,
        _qh: &wayland_client::QueueHandle<Self>,
        _seat: wayland_client::protocol::wl_seat::WlSeat,
        _capability: smithay_client_toolkit::seat::Capability,
    ) {
    }

    fn remove_seat(
        &mut self,
        _conn: &Connection,
        _qh: &wayland_client::QueueHandle<Self>,
        _seat: wayland_client::protocol::wl_seat::WlSeat,
    ) {
    }
}

delegate_compositor!(State);
delegate_output!(State);

delegate_seat!(State);

delegate_xdg_shell!(State);
delegate_xdg_window!(State);

delegate_registry!(State);

impl ProvidesRegistryState for State {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState];
}

fn main() {
    env_logger::init();

    let conn = Connection::connect_to_env().unwrap();
    let (globals, event_queue) = registry_queue_init(&conn).unwrap();
    let qh = event_queue.handle();
    let mut event_loop: EventLoop<State> =
        EventLoop::try_new().expect("failed to initialize event loop");
    let loop_handle = event_loop.handle();
    WaylandSource::new(conn.clone(), event_queue)
        .insert(loop_handle)
        .unwrap();

    let compositor_state = CompositorState::bind(&globals, &qh).expect("xdg shell not available");
    let xdg_shell_state = XdgShell::bind(&globals, &qh).expect("xdg shell not available");

    let surface = compositor_state.create_surface(&qh);
    let window = xdg_shell_state.create_window(
        surface,
        smithay_client_toolkit::shell::xdg::window::WindowDecorations::ServerDefault,
        &qh,
    );
    window.set_title("Vulkan OpenGL Demo");

    window.set_app_id("io.github.euclio.vulkan-opengl-demo.Demo");
    window.set_min_size(Some((256, 256)));
    window.commit();

    let hal_instance = unsafe {
        Instance::init_with_callback(
            &wgpu_hal::InstanceDescriptor {
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

    let raw_display_handle = RawDisplayHandle::Wayland(WaylandDisplayHandle::new(
        NonNull::new(conn.backend().display_id().as_ptr().cast()).unwrap(),
    ));
    let raw_window_handle = RawWindowHandle::Wayland(WaylandWindowHandle::new(
        NonNull::new(window.wl_surface().id().as_ptr().cast()).unwrap(),
    ));

    let surface = unsafe {
        instance
            .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(raw_display_handle),
                raw_window_handle,
            })
            .unwrap()
    };

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&surface),
        ..Default::default()
    }))
    .expect("Failed to find a suitable adapter");

    let surface_capabilities = surface.get_capabilities(&adapter);

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
        adapter.create_device_from_hal::<Vulkan>(open_device, &wgpu::DeviceDescriptor::default())
    }
    .expect("Failed to request device");

    let hal_device = unsafe { device.as_hal::<Vulkan>().unwrap() };
    let raw_vk_device = hal_device.raw_device();

    let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("F texture"),
        size: wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
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
                format: surface_capabilities.formats[0],
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

    let egl_display = unsafe { EGL.get_display(conn.backend().display_ptr().cast()) }.unwrap();
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

    let ext_sem_fn = ash::khr::external_semaphore_fd::Device::new(raw_vk_instance, raw_vk_device);
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
    let raw_queue = hal_queue.as_raw();

    // Pre-signal so GL can proceed on frame 0
    unsafe {
        raw_vk_device
            .queue_submit(
                raw_queue,
                &[vk::SubmitInfo::default().signal_semaphores(&[vk_read_finished])],
                vk::Fence::null(),
            )
            .unwrap();
    }

    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();

    let semaphores = Semaphores {
        gl_write_finished,
        gl_gl_write_finished,
        vk_read_finished,
        gl_vk_read_finished,
    };

    let mut state = State {
        loop_handle: event_loop.handle(),
        registry_state: RegistryState::new(&globals),
        seat_state: SeatState::new(&globals, &qh),
        output_state: OutputState::new(&globals, &qh),
        configured: false,

        exit: false,
        width: 256,
        height: 256,
        window,
        instance,
        device,
        surface,
        adapter,
        queue,
        render_pipeline,
        sampler,
        bind_group_layout,
        bind_group,

        egl_display,
        shared_texture: None,
        semaphores,
    };

    loop {
        event_loop
            .dispatch(Duration::from_millis(16), &mut state)
            .unwrap();

        if state.exit {
            println!("exiting");
            break;
        }
    }

    drop(state.surface);
    drop(state.window);
}
