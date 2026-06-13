# Vulkan/OpenGL Interop Demo

This repository is a demonstration of using OpenGL (compatibility profile) to
render to a Vulkan texture, which is then displayed in a Wayland window.

It is based on the [gl_vk_simple_interop] demo.

Although this demo relies on Vulkan specifically, the pipeline uses [wgpu] where
possible. This allows extending this demo with rendering libraries that
integrate with wgpu.

## Implementation Strategy

When the surface is reconfigured:

1. Create a Vulkan RGBA image.
1. Allocate memory for the image and bind it.
1. Create an opaque file descriptor that describes the memory.
1. Create a GL memory object backed by the file descriptor.
1. Create a GL texture bound to the imported memory.
1. Create a framebuffer bound to the GL texture.

On render:

1. OpenGL waits for Vulkan to signal a semaphore indicating it has finished
   reading the texture.
1. OpenGL renders a scene (using the fixed function pipeline to demonstrate
   compatibility with legacy OpenGL APIs).
1. OpenGL signals Vulkan that writing has completed.
1. Vulkan renders the shared texture to a fullscreen quad.

[gl_vk_simple_interop]: https://github.com/nvpro-samples/gl_vk_simple_interop
[wgpu]: https://github.com/gfx-rs/wgpu
