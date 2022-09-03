//#![allow(unused, dead_code)]
use image::{ImageBuffer, Rgba};
use vulkano::format::Format;
use vulkano::pipeline::graphics::viewport::Scissor;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, RenderPassBeginInfo,
        SubpassContents,
    },
    device::{physical::PhysicalDevice, Device, DeviceCreateInfo, QueueCreateInfo},
    image::{view::ImageView, ImageDimensions, StorageImage},
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, Subpass},
    sync::{self, GpuFuture},
};

mod vertex_shader {
    vulkano_shaders::shader!(
        ty: "vertex",
        src: "
#version 460

layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;

layout(location = 1) out vec4 COLOR;

void main() {
    vec2 translation = vec2(float(gl_InstanceIndex))/10;
    vec2 scale = vec2(float(gl_InstanceIndex))/10;
    
    gl_Position = vec4(position*scale+translation, 0.0, 1.0);
    COLOR = color;
    COLOR.xyz += vec3(float(gl_InstanceIndex)/10);
}
        "
    );
}

mod fragment_shader {
    vulkano_shaders::shader!(
        ty: "fragment",
        src: "
#version 460

layout(location = 1) in vec4 COLOR;

layout(location = 0) out vec4 f_color;

void main() {
    f_color = COLOR;
}
        "
    );
}

type VertexPosition = [f32; 2];
type VertexColor = [f32; 4];

#[repr(C)]
#[derive(bytemuck::Zeroable, bytemuck::Pod, Copy, Clone, Default)]
struct Vertex {
    position: VertexPosition,
    color: VertexColor,
}

impl Vertex {
    fn new(position: VertexPosition, color: VertexColor) -> Self {
        Self { position, color }
    }
}

vulkano::impl_vertex!(Vertex, position, color);

fn main() {
    let resolution = (1024, 1024);
    let format = Format::R8G8B8A8_UNORM;
    let background_color = [0.1, 0.1, 0.1, 1.0]; // RGBA 0-1
    let format_components_count = format.components().len() as u32;

    // How much triangles to draw
    let instance_count = 10;
    let output_filename = "image.png";

    let vertices = vec![
        Vertex::new([-0.5, -0.5], [1.0, 0.0, 0.0, 1.0]),
        Vertex::new([0.0, 0.5], [0.0, 1.0, 0.0, 1.0]),
        Vertex::new([0.5, -0.25], [0.0, 0.0, 1.0, 1.0]),
        Vertex::new([0.0, -1.0], [0.0, 1.0, 0.0, 1.0]),
    ];

    let indices: Vec<u32> = vec![
        0, 1, 2, // triangle 0
        0, 2, 3, // triangle 1
    ];

    // How much vertices to draw
    let first_index = 0;
    let index_count = indices.len() as u32;

    // Setup
    let instance = Instance::new(InstanceCreateInfo::default()).unwrap();
    let physical_device = PhysicalDevice::enumerate(&instance).next().unwrap();
    let queue_family = physical_device
        .queue_families()
        .find(|&queue_family| queue_family.supports_graphics())
        .unwrap();
    let (logical_device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            // here we pass the desired queue families that we want to use
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    // Creating image
    let image = StorageImage::new(
        logical_device.clone(),
        ImageDimensions::Dim2d {
            width: resolution.0,
            height: resolution.1,
            array_layers: 1,
        },
        format,
        Some(queue.family()),
    )
    .unwrap();

    // Create buffers
    let image_pixels_count = resolution.0 * resolution.1;
    let image_components_count = image_pixels_count * format_components_count;

    let image_buffer = CpuAccessibleBuffer::from_iter(
        logical_device.clone(),
        BufferUsage::all(),
        false,
        (0..image_components_count).map(|_| format.components()[0]),
    )
    .unwrap();

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        logical_device.clone(),
        BufferUsage::vertex_buffer(),
        false,
        vertices.into_iter(),
    )
    .unwrap();

    let index_buffer = CpuAccessibleBuffer::from_iter(
        logical_device.clone(),
        BufferUsage::index_buffer(),
        false,
        indices.into_iter(),
    )
    .unwrap();

    // Create a renderpass
    let render_pass = vulkano::single_pass_renderpass!(logical_device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: format,
                samples: 1, // MSAA
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap();

    // Create a framebuffer
    let view = ImageView::new_default(image.clone()).unwrap();
    let framebuffer = Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![view],
            ..Default::default()
        },
    )
    .unwrap();

    // Load shaders
    let loaded_vertex_shader = vertex_shader::load(logical_device.clone()).unwrap();
    let loaded_fragment_shader = fragment_shader::load(logical_device.clone()).unwrap();

    // Setup the static viewport
    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [resolution.0 as f32, resolution.1 as f32],
        depth_range: 0.0..1.0,
    };

    let scissor = Scissor {
        origin: [0, 0],//[resolution.0/2, resolution.1/2],
        dimensions: [resolution.0, resolution.1]//[resolution.0/4, resolution.1/4],
    };

    // Creating a graphics pipeline
    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(loaded_vertex_shader.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_fixed([(viewport, scissor)]))
        .fragment_shader(loaded_fragment_shader.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(logical_device.clone())
        .unwrap();

    // Create a command buffer
    let mut builder = AutoCommandBufferBuilder::primary(
        logical_device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some(background_color.into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassContents::Inline,
        )
        .unwrap()
        .bind_pipeline_graphics(pipeline.clone())
        .bind_vertex_buffers(0, vertex_buffer.clone())
        .bind_index_buffer(index_buffer)
        .draw_indexed(index_count, instance_count, first_index, 0, 0)
        .unwrap()
        .end_render_pass()
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image, image_buffer.clone()))
        .unwrap();

    let command_buffer = builder.build().unwrap();

    // Creating a future
    let future = sync::now(logical_device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    // Waiting for the logical device to sync, upload the command buffer and execute it
    future.wait(None).unwrap();

    // Saving rendered image
    let buffer_content = image_buffer.read().unwrap();
    let image =
        ImageBuffer::<Rgba<u8>, _>::from_raw(resolution.0, resolution.1, &buffer_content[..])
            .unwrap();
    image.save(output_filename).unwrap();

    println!("Everything succeeded!");
}
