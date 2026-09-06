use std::{
    ffi::{CStr, CString},
    path::Path,
};

use ash::{Entry, vk};

mod compute_graph;

pub use compute_graph::{
    AccessType, ComputeGraph, ComputeGraphDefinition, ComputeGraphError, ComputeGraphExecution,
    ComputeNodeDefinition, ResourceBindingDefinition, ResourceDefinition, ResourceKind,
};

#[derive(Debug)]
pub struct PhysicalDeviceInfo {
    pub name: String,
    pub device_type: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub api_version: (u32, u32, u32),
    pub driver_version: u32,
    pub queue_family_count: usize,
}

pub fn enumerate_physical_devices() -> Result<Vec<PhysicalDeviceInfo>, Box<dyn std::error::Error>> {
    let entry = unsafe { Entry::load()? };
    let app_name = CString::new("ai-vk")?;
    let engine_name = CString::new("ai-vk")?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_0);
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_info, None)? };

    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    let devices = physical_devices
        .iter()
        .map(|physical_device| {
            let properties = unsafe { instance.get_physical_device_properties(*physical_device) };
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };
            let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            let api_version = properties.api_version;

            PhysicalDeviceInfo {
                name,
                device_type: format!("{:?}", properties.device_type),
                vendor_id: properties.vendor_id,
                device_id: properties.device_id,
                api_version: (
                    vk::api_version_major(api_version),
                    vk::api_version_minor(api_version),
                    vk::api_version_patch(api_version),
                ),
                driver_version: properties.driver_version,
                queue_family_count: queue_families.len(),
            }
        })
        .collect();

    unsafe { instance.destroy_instance(None) };
    Ok(devices)
}

/// Creates an image, clears it using a compute queue, and writes it as a PNG.
///
/// No graphics queue is created or used. The image is copied to a host-visible
/// buffer only after the GPU clear has completed so the `image` crate can encode it.
pub fn write_cleared_image_png(
    width: u32,
    height: u32,
    color: [u8; 4],
    output_path: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    if width == 0 || height == 0 {
        return Err("image dimensions must be greater than zero".into());
    }

    let pixel_count = width
        .checked_mul(height)
        .ok_or("image dimensions overflow")?;
    let byte_count = pixel_count.checked_mul(4).ok_or("image size overflow")?;
    let byte_count = vk::DeviceSize::from(byte_count);

    let entry = unsafe { Entry::load()? };
    let app_name = CString::new("ai-vk")?;
    let engine_name = CString::new("ai-vk")?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_0);
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_info, None)? };

    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    let (physical_device, queue_family_index) = select_compute_queue(&instance, &physical_devices)?;
    let queue_priority = [1.0_f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priority);
    let device_info =
        vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_info));
    let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
    let mut resources = ComputeResources {
        instance,
        device,
        image_table: None,
        sampler_table: None,
        image: vk::Image::null(),
        image_view: vk::ImageView::null(),
        image_memory: vk::DeviceMemory::null(),
        source_image: vk::Image::null(),
        source_image_view: vk::ImageView::null(),
        source_image_memory: vk::DeviceMemory::null(),
        buffer: vk::Buffer::null(),
        buffer_memory: vk::DeviceMemory::null(),
        command_pool: vk::CommandPool::null(),
        shader_module: vk::ShaderModule::null(),
        downsample_shader_module: vk::ShaderModule::null(),
        pipeline_layout: vk::PipelineLayout::null(),
        pipeline: vk::Pipeline::null(),
        downsample_pipeline: vk::Pipeline::null(),
    };
    let queue = unsafe { resources.device.get_device_queue(queue_family_index, 0) };

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    resources.image = unsafe { resources.device.create_image(&image_info, None)? };
    let image_requirements = unsafe {
        resources
            .device
            .get_image_memory_requirements(resources.image)
    };
    let memory_properties = unsafe {
        resources
            .instance
            .get_physical_device_memory_properties(physical_device)
    };
    let (image_memory_type, _) = find_memory_type(
        &memory_properties,
        image_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
    )
    .or_else(|_| {
        find_memory_type(
            &memory_properties,
            image_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::empty(),
            vk::MemoryPropertyFlags::empty(),
        )
    })?;
    let image_memory_info = vk::MemoryAllocateInfo::default()
        .allocation_size(image_requirements.size)
        .memory_type_index(image_memory_type);
    resources.image_memory = unsafe { resources.device.allocate_memory(&image_memory_info, None)? };
    unsafe {
        resources
            .device
            .bind_image_memory(resources.image, resources.image_memory, 0)?;
    }

    let buffer_info = vk::BufferCreateInfo::default()
        .size(byte_count)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    resources.buffer = unsafe { resources.device.create_buffer(&buffer_info, None)? };
    let buffer_requirements = unsafe {
        resources
            .device
            .get_buffer_memory_requirements(resources.buffer)
    };
    let (buffer_memory_type, buffer_memory_flags) = find_memory_type(
        &memory_properties,
        buffer_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
        vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let buffer_memory_info = vk::MemoryAllocateInfo::default()
        .allocation_size(buffer_requirements.size)
        .memory_type_index(buffer_memory_type);
    resources.buffer_memory = unsafe {
        resources
            .device
            .allocate_memory(&buffer_memory_info, None)?
    };
    unsafe {
        resources
            .device
            .bind_buffer_memory(resources.buffer, resources.buffer_memory, 0)?;
    }

    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(queue_family_index);
    resources.command_pool = unsafe {
        resources
            .device
            .create_command_pool(&command_pool_info, None)?
    };
    let command_buffer_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(resources.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = unsafe {
        resources
            .device
            .allocate_command_buffers(&command_buffer_info)?[0]
    };

    let begin_info = vk::CommandBufferBeginInfo::default();
    unsafe {
        resources
            .device
            .begin_command_buffer(command_buffer, &begin_info)?;

        let to_transfer_dst = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(resources.image)
            .subresource_range(color_subresource_range());
        resources.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&to_transfer_dst),
        );

        let clear_color = vk::ClearColorValue {
            float32: [
                f32::from(color[0]) / 255.0,
                f32::from(color[1]) / 255.0,
                f32::from(color[2]) / 255.0,
                f32::from(color[3]) / 255.0,
            ],
        };
        resources.device.cmd_clear_color_image(
            command_buffer,
            resources.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &clear_color,
            std::slice::from_ref(&color_subresource_range()),
        );

        let to_transfer_src = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(resources.image)
            .subresource_range(color_subresource_range());
        resources.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&to_transfer_src),
        );

        let copy_region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        resources.device.cmd_copy_image_to_buffer(
            command_buffer,
            resources.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            resources.buffer,
            std::slice::from_ref(&copy_region),
        );
        resources.device.end_command_buffer(command_buffer)?;

        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
        resources.device.queue_submit(
            queue,
            std::slice::from_ref(&submit_info),
            vk::Fence::null(),
        )?;
        resources.device.queue_wait_idle(queue)?;
    }

    let mapped_memory = unsafe {
        resources.device.map_memory(
            resources.buffer_memory,
            0,
            byte_count,
            vk::MemoryMapFlags::empty(),
        )?
    };
    if !buffer_memory_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
        let range = vk::MappedMemoryRange::default()
            .memory(resources.buffer_memory)
            .offset(0)
            .size(byte_count);
        unsafe {
            resources
                .device
                .invalidate_mapped_memory_ranges(std::slice::from_ref(&range))?;
        }
    }
    let pixels =
        unsafe { std::slice::from_raw_parts(mapped_memory.cast::<u8>(), byte_count as usize) };
    let pixels = pixels.to_vec();
    unsafe { resources.device.unmap_memory(resources.buffer_memory) };

    let image = image::RgbaImage::from_raw(width, height, pixels)
        .ok_or("GPU image data did not match the requested dimensions")?;
    image.save_with_format(output_path, image::ImageFormat::Png)?;

    Ok(())
}

/// Runs a compute shader that writes a solid color to an image selected from
/// the bindless image table, then writes the image as a PNG.
///
/// The shader must have a `main` compute entry point and use the descriptor
/// layout declared by `shaders/bindless_images.hlsl`.
pub fn write_bindless_image_png(
    width: u32,
    height: u32,
    color: [u8; 4],
    shader_spirv: &[u8],
    output_path: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    if width == 0 || height == 0 {
        return Err("image dimensions must be greater than zero".into());
    }
    if !shader_spirv
        .len()
        .is_multiple_of(std::mem::size_of::<u32>())
    {
        return Err("compute shader SPIR-V size must be a multiple of four".into());
    }

    let pixel_count = width
        .checked_mul(height)
        .ok_or("image dimensions overflow")?;
    let byte_count = vk::DeviceSize::from(pixel_count.checked_mul(4).ok_or("image size overflow")?);

    let entry = unsafe { Entry::load()? };
    let app_name = CString::new("ai-vk")?;
    let engine_name = CString::new("ai-vk")?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_2);
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_info, None)? };

    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    let (physical_device, queue_family_index) = select_compute_queue(&instance, &physical_devices)?;
    let (core_features_supported, partially_bound_supported) = {
        let mut descriptor_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
        let mut supported_features =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut descriptor_features);
        unsafe {
            instance.get_physical_device_features2(physical_device, &mut supported_features);
        }
        (
            supported_features.features,
            descriptor_features.descriptor_binding_partially_bound,
        )
    };
    if core_features_supported.shader_storage_image_array_dynamic_indexing == vk::FALSE {
        return Err(
            "selected Vulkan device does not support dynamic storage-image indexing".into(),
        );
    }
    if partially_bound_supported == vk::FALSE {
        return Err("selected Vulkan device does not support partially bound descriptors".into());
    }
    let core_features = vk::PhysicalDeviceFeatures {
        shader_storage_image_array_dynamic_indexing: vk::TRUE,
        ..Default::default()
    };
    let mut descriptor_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
        .descriptor_binding_partially_bound(true);

    let queue_priority = [1.0_f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priority);
    let mut device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .enabled_features(&core_features);
    device_info = device_info.push_next(&mut descriptor_features);
    let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
    let mut resources = ComputeResources::new(instance, device)?;

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    resources.image = unsafe { resources.device.create_image(&image_info, None)? };
    let image_requirements = unsafe {
        resources
            .device
            .get_image_memory_requirements(resources.image)
    };
    let memory_properties = unsafe {
        resources
            .instance
            .get_physical_device_memory_properties(physical_device)
    };
    let (image_memory_type, _) = find_memory_type(
        &memory_properties,
        image_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
    )
    .or_else(|_| {
        find_memory_type(
            &memory_properties,
            image_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::empty(),
            vk::MemoryPropertyFlags::empty(),
        )
    })?;
    let image_memory_info = vk::MemoryAllocateInfo::default()
        .allocation_size(image_requirements.size)
        .memory_type_index(image_memory_type);
    resources.image_memory = unsafe { resources.device.allocate_memory(&image_memory_info, None)? };
    unsafe {
        resources
            .device
            .bind_image_memory(resources.image, resources.image_memory, 0)?;
    }

    let image_view_info = vk::ImageViewCreateInfo::default()
        .image(resources.image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .subresource_range(color_subresource_range());
    resources.image_view = unsafe { resources.device.create_image_view(&image_view_info, None)? };
    let image_index = resources
        .image_table
        .as_mut()
        .expect("bindless image table should be present")
        .add_image(&resources.device, resources.image_view)
        .map_err(|error| format!("could not add image to bindless table: {error}"))?;

    let buffer_info = vk::BufferCreateInfo::default()
        .size(byte_count)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    resources.buffer = unsafe { resources.device.create_buffer(&buffer_info, None)? };
    let buffer_requirements = unsafe {
        resources
            .device
            .get_buffer_memory_requirements(resources.buffer)
    };
    let (buffer_memory_type, buffer_memory_flags) = find_memory_type(
        &memory_properties,
        buffer_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
        vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let buffer_memory_info = vk::MemoryAllocateInfo::default()
        .allocation_size(buffer_requirements.size)
        .memory_type_index(buffer_memory_type);
    resources.buffer_memory = unsafe {
        resources
            .device
            .allocate_memory(&buffer_memory_info, None)?
    };
    unsafe {
        resources
            .device
            .bind_buffer_memory(resources.buffer, resources.buffer_memory, 0)?;
    }

    let shader_code: Vec<u32> = shader_spirv
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect();
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
    resources.shader_module = unsafe { resources.device.create_shader_module(&shader_info, None)? };
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<BindlessPushConstants>() as u32);
    let set_layouts = [
        resources
            .sampler_table
            .as_ref()
            .expect("immutable sampler table should be present")
            .layout,
        resources
            .image_table
            .as_ref()
            .expect("bindless image table should be present")
            .layout,
    ];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));
    resources.pipeline_layout = unsafe {
        resources
            .device
            .create_pipeline_layout(&pipeline_layout_info, None)?
    };
    let entry_point = CString::new("main")?;
    let shader_stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(resources.shader_module)
        .name(&entry_point);
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(shader_stage)
        .layout(resources.pipeline_layout);
    resources.pipeline = unsafe {
        resources
            .device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            )
            .map_err(|(_, error)| error)?[0]
    };

    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(queue_family_index);
    resources.command_pool = unsafe {
        resources
            .device
            .create_command_pool(&command_pool_info, None)?
    };
    let command_buffer_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(resources.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = unsafe {
        resources
            .device
            .allocate_command_buffers(&command_buffer_info)?[0]
    };
    let push_constants = BindlessPushConstants {
        target_and_extent: [image_index, width, height, 0],
        color: color.map(|channel| f32::from(channel) / 255.0),
    };
    let push_constant_bytes = unsafe {
        std::slice::from_raw_parts(
            (&push_constants as *const BindlessPushConstants).cast::<u8>(),
            std::mem::size_of::<BindlessPushConstants>(),
        )
    };

    unsafe {
        resources
            .device
            .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;
        let to_general = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(resources.image)
            .subresource_range(color_subresource_range());
        resources.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&to_general),
        );
        resources.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            resources.pipeline,
        );
        resources.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            resources.pipeline_layout,
            0,
            &[
                resources
                    .sampler_table
                    .as_ref()
                    .expect("immutable sampler table should be present")
                    .set,
                resources
                    .image_table
                    .as_ref()
                    .expect("bindless image table should be present")
                    .set,
            ],
            &[],
        );
        resources.device.cmd_push_constants(
            command_buffer,
            resources.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            push_constant_bytes,
        );
        resources
            .device
            .cmd_dispatch(command_buffer, width.div_ceil(8), height.div_ceil(8), 1);

        let to_transfer_src = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(resources.image)
            .subresource_range(color_subresource_range());
        resources.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&to_transfer_src),
        );
        let copy_region = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        resources.device.cmd_copy_image_to_buffer(
            command_buffer,
            resources.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            resources.buffer,
            std::slice::from_ref(&copy_region),
        );
        resources.device.end_command_buffer(command_buffer)?;
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
        resources.device.queue_submit(
            queue,
            std::slice::from_ref(&submit_info),
            vk::Fence::null(),
        )?;
        resources.device.queue_wait_idle(queue)?;
    }

    let mapped_memory = unsafe {
        resources.device.map_memory(
            resources.buffer_memory,
            0,
            byte_count,
            vk::MemoryMapFlags::empty(),
        )?
    };
    if !buffer_memory_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
        let range = vk::MappedMemoryRange::default()
            .memory(resources.buffer_memory)
            .offset(0)
            .size(byte_count);
        unsafe {
            resources
                .device
                .invalidate_mapped_memory_ranges(std::slice::from_ref(&range))?
        };
    }
    let pixels = unsafe {
        std::slice::from_raw_parts(mapped_memory.cast::<u8>(), byte_count as usize).to_vec()
    };
    unsafe { resources.device.unmap_memory(resources.buffer_memory) };

    let image = image::RgbaImage::from_raw(width, height, pixels)
        .ok_or("GPU image data did not match the requested dimensions")?;
    image.save_with_format(output_path, image::ImageFormat::Png)?;
    Ok(())
}

/// Runs the SDF shader into an intermediate image, then downsamples that image
/// with a sampled-image shader and a fixed linear sampler.
pub fn write_sdf_downsampled_image_png(
    width: u32,
    height: u32,
    sdf_shader_spirv: &[u8],
    downsample_shader_spirv: &[u8],
    output_path: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    if width < 2 || height < 2 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err("SDF dimensions must be even and at least two pixels".into());
    }
    for shader_spirv in [sdf_shader_spirv, downsample_shader_spirv] {
        if !shader_spirv
            .len()
            .is_multiple_of(std::mem::size_of::<u32>())
        {
            return Err("compute shader SPIR-V size must be a multiple of four".into());
        }
    }

    let output_width = width / 2;
    let output_height = height / 2;
    let pixel_count = output_width
        .checked_mul(output_height)
        .ok_or("image dimensions overflow")?;
    let byte_count = vk::DeviceSize::from(pixel_count.checked_mul(4).ok_or("image size overflow")?);

    let entry = unsafe { Entry::load()? };
    let app_name = CString::new("ai-vk")?;
    let engine_name = CString::new("ai-vk")?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_2);
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_info, None)? };

    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    let (physical_device, queue_family_index) = select_compute_queue(&instance, &physical_devices)?;
    let (core_features_supported, partially_bound_supported) = {
        let mut descriptor_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
        let mut supported_features =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut descriptor_features);
        unsafe {
            instance.get_physical_device_features2(physical_device, &mut supported_features);
        }
        (
            supported_features.features,
            descriptor_features.descriptor_binding_partially_bound,
        )
    };
    if core_features_supported.shader_storage_image_array_dynamic_indexing == vk::FALSE {
        return Err(
            "selected Vulkan device does not support dynamic storage-image indexing".into(),
        );
    }
    if core_features_supported.shader_sampled_image_array_dynamic_indexing == vk::FALSE {
        return Err(
            "selected Vulkan device does not support dynamic sampled-image indexing".into(),
        );
    }
    if partially_bound_supported == vk::FALSE {
        return Err("selected Vulkan device does not support partially bound descriptors".into());
    }
    let core_features = vk::PhysicalDeviceFeatures {
        shader_storage_image_array_dynamic_indexing: vk::TRUE,
        shader_sampled_image_array_dynamic_indexing: vk::TRUE,
        ..Default::default()
    };
    let mut descriptor_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
        .descriptor_binding_partially_bound(true);

    let queue_priority = [1.0_f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priority);
    let mut device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .enabled_features(&core_features);
    device_info = device_info.push_next(&mut descriptor_features);
    let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
    let mut resources = ComputeResources::new(instance, device)?;
    let memory_properties = unsafe {
        resources
            .instance
            .get_physical_device_memory_properties(physical_device)
    };

    let source_image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let (source_image, source_memory) =
        create_device_image(&resources.device, &source_image_info, &memory_properties)?;
    resources.source_image = source_image;
    resources.source_image_memory = source_memory;
    let source_view_info = vk::ImageViewCreateInfo::default()
        .image(resources.source_image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .subresource_range(color_subresource_range());
    resources.source_image_view = unsafe {
        resources
            .device
            .create_image_view(&source_view_info, None)?
    };

    let output_image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent3D {
            width: output_width,
            height: output_height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let (output_image, output_memory) =
        create_device_image(&resources.device, &output_image_info, &memory_properties)?;
    resources.image = output_image;
    resources.image_memory = output_memory;
    let output_view_info = vk::ImageViewCreateInfo::default()
        .image(resources.image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .subresource_range(color_subresource_range());
    resources.image_view = unsafe {
        resources
            .device
            .create_image_view(&output_view_info, None)?
    };

    let source_image_index = resources
        .image_table
        .as_mut()
        .expect("bindless image table should be present")
        .add_image(&resources.device, resources.source_image_view)
        .map_err(|error| format!("could not add source image to bindless table: {error}"))?;
    let output_image_index = resources
        .image_table
        .as_mut()
        .expect("bindless image table should be present")
        .add_image(&resources.device, resources.image_view)
        .map_err(|error| format!("could not add output image to bindless table: {error}"))?;
    let source_texture_index = resources
        .image_table
        .as_mut()
        .expect("bindless image table should be present")
        .add_texture(&resources.device, resources.source_image_view)
        .map_err(|error| format!("could not add source texture to bindless table: {error}"))?;

    let buffer_info = vk::BufferCreateInfo::default()
        .size(byte_count)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    resources.buffer = unsafe { resources.device.create_buffer(&buffer_info, None)? };
    let buffer_requirements = unsafe {
        resources
            .device
            .get_buffer_memory_requirements(resources.buffer)
    };
    let (buffer_memory_type, buffer_memory_flags) = find_memory_type(
        &memory_properties,
        buffer_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
        vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let buffer_memory_info = vk::MemoryAllocateInfo::default()
        .allocation_size(buffer_requirements.size)
        .memory_type_index(buffer_memory_type);
    resources.buffer_memory = unsafe {
        resources
            .device
            .allocate_memory(&buffer_memory_info, None)?
    };
    unsafe {
        resources
            .device
            .bind_buffer_memory(resources.buffer, resources.buffer_memory, 0)?;
    }

    let sdf_shader_code: Vec<u32> = sdf_shader_spirv
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect();
    let downsample_shader_code: Vec<u32> = downsample_shader_spirv
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect();
    resources.shader_module = unsafe {
        resources.device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&sdf_shader_code),
            None,
        )?
    };
    resources.downsample_shader_module = unsafe {
        resources.device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&downsample_shader_code),
            None,
        )?
    };
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<BindlessPushConstants>() as u32);
    let set_layouts = [
        resources
            .sampler_table
            .as_ref()
            .expect("immutable sampler table should be present")
            .layout,
        resources
            .image_table
            .as_ref()
            .expect("bindless image table should be present")
            .layout,
    ];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));
    resources.pipeline_layout = unsafe {
        resources
            .device
            .create_pipeline_layout(&pipeline_layout_info, None)?
    };
    let entry_point = CString::new("main")?;
    let sdf_shader_stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(resources.shader_module)
        .name(&entry_point);
    let sdf_pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(sdf_shader_stage)
        .layout(resources.pipeline_layout);
    resources.pipeline = unsafe {
        resources
            .device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&sdf_pipeline_info),
                None,
            )
            .map_err(|(_, error)| error)?[0]
    };
    let downsample_shader_stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(resources.downsample_shader_module)
        .name(&entry_point);
    let downsample_pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(downsample_shader_stage)
        .layout(resources.pipeline_layout);
    resources.downsample_pipeline = unsafe {
        resources
            .device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&downsample_pipeline_info),
                None,
            )
            .map_err(|(_, error)| error)?[0]
    };

    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(queue_family_index);
    resources.command_pool = unsafe {
        resources
            .device
            .create_command_pool(&command_pool_info, None)?
    };
    let command_buffer_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(resources.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = unsafe {
        resources
            .device
            .allocate_command_buffers(&command_buffer_info)?[0]
    };
    let descriptor_sets = [
        resources
            .sampler_table
            .as_ref()
            .expect("immutable sampler table should be present")
            .set,
        resources
            .image_table
            .as_ref()
            .expect("bindless image table should be present")
            .set,
    ];
    let sdf_push_constants = BindlessPushConstants {
        target_and_extent: [source_image_index, width, height, 0],
        color: [0.0; 4],
    };
    let downsample_push_constants = BindlessPushConstants {
        target_and_extent: [
            output_image_index,
            output_width,
            output_height,
            source_texture_index,
        ],
        color: [0.0; 4],
    };
    let sdf_push_constant_bytes = unsafe {
        std::slice::from_raw_parts(
            (&sdf_push_constants as *const BindlessPushConstants).cast::<u8>(),
            std::mem::size_of::<BindlessPushConstants>(),
        )
    };
    let downsample_push_constant_bytes = unsafe {
        std::slice::from_raw_parts(
            (&downsample_push_constants as *const BindlessPushConstants).cast::<u8>(),
            std::mem::size_of::<BindlessPushConstants>(),
        )
    };

    unsafe {
        resources
            .device
            .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;
        let source_to_general = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(resources.source_image)
            .subresource_range(color_subresource_range());
        let output_to_general = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(resources.image)
            .subresource_range(color_subresource_range());
        resources.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[source_to_general, output_to_general],
        );
        resources.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            resources.pipeline,
        );
        resources.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            resources.pipeline_layout,
            0,
            &descriptor_sets,
            &[],
        );
        resources.device.cmd_push_constants(
            command_buffer,
            resources.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            sdf_push_constant_bytes,
        );
        resources
            .device
            .cmd_dispatch(command_buffer, width.div_ceil(8), height.div_ceil(8), 1);

        let source_to_sampled = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(resources.source_image)
            .subresource_range(color_subresource_range());
        resources.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&source_to_sampled),
        );
        resources.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            resources.downsample_pipeline,
        );
        resources.device.cmd_push_constants(
            command_buffer,
            resources.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            downsample_push_constant_bytes,
        );
        resources.device.cmd_dispatch(
            command_buffer,
            output_width.div_ceil(8),
            output_height.div_ceil(8),
            1,
        );

        let output_to_transfer = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(resources.image)
            .subresource_range(color_subresource_range());
        resources.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&output_to_transfer),
        );
        let copy_region = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: output_width,
                height: output_height,
                depth: 1,
            });
        resources.device.cmd_copy_image_to_buffer(
            command_buffer,
            resources.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            resources.buffer,
            std::slice::from_ref(&copy_region),
        );
        resources.device.end_command_buffer(command_buffer)?;
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
        resources.device.queue_submit(
            queue,
            std::slice::from_ref(&submit_info),
            vk::Fence::null(),
        )?;
        resources.device.queue_wait_idle(queue)?;
    }

    let mapped_memory = unsafe {
        resources.device.map_memory(
            resources.buffer_memory,
            0,
            byte_count,
            vk::MemoryMapFlags::empty(),
        )?
    };
    if !buffer_memory_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
        let range = vk::MappedMemoryRange::default()
            .memory(resources.buffer_memory)
            .offset(0)
            .size(byte_count);
        unsafe {
            resources
                .device
                .invalidate_mapped_memory_ranges(std::slice::from_ref(&range))?
        };
    }
    let pixels = unsafe {
        std::slice::from_raw_parts(mapped_memory.cast::<u8>(), byte_count as usize).to_vec()
    };
    unsafe { resources.device.unmap_memory(resources.buffer_memory) };
    let image = image::RgbaImage::from_raw(output_width, output_height, pixels)
        .ok_or("GPU image data did not match the requested dimensions")?;
    image.save_with_format(output_path, image::ImageFormat::Png)?;
    Ok(())
}

fn create_device_image(
    device: &ash::Device,
    image_info: &vk::ImageCreateInfo<'_>,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
) -> Result<(vk::Image, vk::DeviceMemory), Box<dyn std::error::Error>> {
    let image = unsafe { device.create_image(image_info, None)? };
    let image_requirements = unsafe { device.get_image_memory_requirements(image) };
    let memory_type = match find_memory_type(
        memory_properties,
        image_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
    )
    .or_else(|_| {
        find_memory_type(
            memory_properties,
            image_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::empty(),
            vk::MemoryPropertyFlags::empty(),
        )
    }) {
        Ok((memory_type, _)) => memory_type,
        Err(error) => {
            unsafe { device.destroy_image(image, None) };
            return Err(error);
        }
    };
    let memory_info = vk::MemoryAllocateInfo::default()
        .allocation_size(image_requirements.size)
        .memory_type_index(memory_type);
    let memory = match unsafe { device.allocate_memory(&memory_info, None) } {
        Ok(memory) => memory,
        Err(error) => {
            unsafe { device.destroy_image(image, None) };
            return Err(error.into());
        }
    };
    if let Err(error) = unsafe { device.bind_image_memory(image, memory, 0) } {
        unsafe {
            device.free_memory(memory, None);
            device.destroy_image(image, None);
        }
        return Err(error.into());
    }
    Ok((image, memory))
}

fn select_compute_queue(
    instance: &ash::Instance,
    physical_devices: &[vk::PhysicalDevice],
) -> Result<(vk::PhysicalDevice, u32), Box<dyn std::error::Error>> {
    physical_devices
        .iter()
        .find_map(|physical_device| {
            let families =
                unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };
            families
                .iter()
                .enumerate()
                .find(|(_, family)| {
                    family.queue_count > 0
                        && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                })
                .map(|(index, _)| (*physical_device, index as u32))
        })
        .or_else(|| {
            physical_devices.iter().find_map(|physical_device| {
                let families = unsafe {
                    instance.get_physical_device_queue_family_properties(*physical_device)
                };
                families
                    .iter()
                    .enumerate()
                    .find(|(_, family)| {
                        family.queue_count > 0
                            && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    })
                    .map(|(index, _)| (*physical_device, index as u32))
            })
        })
        .ok_or_else(|| "no compute-capable Vulkan queue was found".into())
}

fn find_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    required: vk::MemoryPropertyFlags,
    preferred: vk::MemoryPropertyFlags,
) -> Result<(u32, vk::MemoryPropertyFlags), Box<dyn std::error::Error>> {
    let mut fallback = None;
    for (index, memory_type) in properties
        .memory_types
        .iter()
        .enumerate()
        .take(properties.memory_type_count as usize)
    {
        if type_filter & (1 << index) == 0 || !memory_type.property_flags.contains(required) {
            continue;
        }
        if memory_type.property_flags.contains(preferred) {
            return Ok((index as u32, memory_type.property_flags));
        }
        fallback = Some((index as u32, memory_type.property_flags));
    }
    fallback.ok_or_else(|| "no compatible Vulkan memory type was found".into())
}

fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}

const BINDLESS_IMAGE_COUNT: u32 = 64;

struct BindlessImageTable {
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    occupied: [bool; BINDLESS_IMAGE_COUNT as usize],
}

struct ImmutableSamplerTable {
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    sampler: vk::Sampler,
}

impl ImmutableSamplerTable {
    fn new(device: &ash::Device) -> Result<Self, Box<dyn std::error::Error>> {
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(0.0);
        let sampler = unsafe { device.create_sampler(&sampler_info, None)? };
        let sampler_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .immutable_samplers(std::slice::from_ref(&sampler));
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(std::slice::from_ref(&sampler_binding));

        let layout = match unsafe { device.create_descriptor_set_layout(&layout_info, None) } {
            Ok(layout) => layout,
            Err(error) => {
                unsafe { device.destroy_sampler(sampler, None) };
                return Err(error.into());
            }
        };

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLER)
            .descriptor_count(1)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let pool = match unsafe { device.create_descriptor_pool(&pool_info, None) } {
            Ok(pool) => pool,
            Err(error) => {
                unsafe {
                    device.destroy_descriptor_set_layout(layout, None);
                    device.destroy_sampler(sampler, None);
                }
                return Err(error.into());
            }
        };
        let set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(std::slice::from_ref(&layout));
        let set = match unsafe { device.allocate_descriptor_sets(&set_info) } {
            Ok(sets) => sets[0],
            Err(error) => {
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                    device.destroy_descriptor_set_layout(layout, None);
                    device.destroy_sampler(sampler, None);
                }
                return Err(error.into());
            }
        };

        Ok(Self {
            layout,
            pool,
            set,
            sampler,
        })
    }
}

impl BindlessImageTable {
    fn new(device: &ash::Device) -> Result<Self, Box<dyn std::error::Error>> {
        let storage_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(BINDLESS_IMAGE_COUNT)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let sampled_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(BINDLESS_IMAGE_COUNT)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let bindings = [storage_binding, sampled_binding];
        let binding_flag_values = [
            vk::DescriptorBindingFlags::PARTIALLY_BOUND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        ];
        let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
            .binding_flags(&binding_flag_values);
        let mut layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        layout_info = layout_info.push_next(&mut binding_flags);
        let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(BINDLESS_IMAGE_COUNT),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(BINDLESS_IMAGE_COUNT),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let pool = match unsafe { device.create_descriptor_pool(&pool_info, None) } {
            Ok(pool) => pool,
            Err(error) => {
                unsafe { device.destroy_descriptor_set_layout(layout, None) };
                return Err(error.into());
            }
        };
        let set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(std::slice::from_ref(&layout));
        let set = match unsafe { device.allocate_descriptor_sets(&set_info) } {
            Ok(sets) => sets[0],
            Err(error) => {
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                    device.destroy_descriptor_set_layout(layout, None);
                }
                return Err(error.into());
            }
        };

        Ok(Self {
            layout,
            pool,
            set,
            occupied: [false; BINDLESS_IMAGE_COUNT as usize],
        })
    }

    fn add_image(
        &mut self,
        device: &ash::Device,
        image_view: vk::ImageView,
    ) -> Result<u32, &'static str> {
        let index = self
            .occupied
            .iter()
            .position(|occupied| !occupied)
            .ok_or("bindless image table is full")?;
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(image_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(0)
            .dst_array_element(index as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&image_info));
        // The descriptor remains in this set for the lifetime of the image.
        // No per-dispatch descriptor writes are needed.
        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        self.occupied[index] = true;
        Ok(index as u32)
    }

    fn add_texture(
        &mut self,
        device: &ash::Device,
        image_view: vk::ImageView,
    ) -> Result<u32, &'static str> {
        let index = self
            .occupied
            .iter()
            .position(|occupied| !occupied)
            .ok_or("bindless texture table is full")?;
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(1)
            .dst_array_element(index as u32)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&image_info));
        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        self.occupied[index] = true;
        Ok(index as u32)
    }
}

#[repr(C)]
struct BindlessPushConstants {
    target_and_extent: [u32; 4],
    color: [f32; 4],
}

struct ComputeResources {
    instance: ash::Instance,
    device: ash::Device,
    image_table: Option<BindlessImageTable>,
    sampler_table: Option<ImmutableSamplerTable>,
    image: vk::Image,
    image_view: vk::ImageView,
    image_memory: vk::DeviceMemory,
    source_image: vk::Image,
    source_image_view: vk::ImageView,
    source_image_memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    command_pool: vk::CommandPool,
    shader_module: vk::ShaderModule,
    downsample_shader_module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    downsample_pipeline: vk::Pipeline,
}

impl ComputeResources {
    fn new(
        instance: ash::Instance,
        device: ash::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let image_table = BindlessImageTable::new(&device)?;
        let sampler_table = match ImmutableSamplerTable::new(&device) {
            Ok(sampler_table) => sampler_table,
            Err(error) => {
                unsafe {
                    device.destroy_descriptor_pool(image_table.pool, None);
                    device.destroy_descriptor_set_layout(image_table.layout, None);
                }
                return Err(error);
            }
        };
        Ok(Self {
            instance,
            device,
            image_table: Some(image_table),
            sampler_table: Some(sampler_table),
            image: vk::Image::null(),
            image_view: vk::ImageView::null(),
            image_memory: vk::DeviceMemory::null(),
            source_image: vk::Image::null(),
            source_image_view: vk::ImageView::null(),
            source_image_memory: vk::DeviceMemory::null(),
            buffer: vk::Buffer::null(),
            buffer_memory: vk::DeviceMemory::null(),
            command_pool: vk::CommandPool::null(),
            shader_module: vk::ShaderModule::null(),
            downsample_shader_module: vk::ShaderModule::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            downsample_pipeline: vk::Pipeline::null(),
        })
    }
}

impl Drop for ComputeResources {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if self.command_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.command_pool, None);
            }
            if self.buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.buffer, None);
            }
            if self.buffer_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.buffer_memory, None);
            }
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
            }
            if self.downsample_pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.downsample_pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.shader_module != vk::ShaderModule::null() {
                self.device.destroy_shader_module(self.shader_module, None);
            }
            if self.downsample_shader_module != vk::ShaderModule::null() {
                self.device
                    .destroy_shader_module(self.downsample_shader_module, None);
            }
            if let Some(image_table) = self.image_table.take() {
                self.device.destroy_descriptor_pool(image_table.pool, None);
                self.device
                    .destroy_descriptor_set_layout(image_table.layout, None);
            }
            if let Some(sampler_table) = self.sampler_table.take() {
                self.device
                    .destroy_descriptor_pool(sampler_table.pool, None);
                self.device
                    .destroy_descriptor_set_layout(sampler_table.layout, None);
                self.device.destroy_sampler(sampler_table.sampler, None);
            }
            if self.image_view != vk::ImageView::null() {
                self.device.destroy_image_view(self.image_view, None);
            }
            if self.image != vk::Image::null() {
                self.device.destroy_image(self.image, None);
            }
            if self.image_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.image_memory, None);
            }
            if self.source_image_view != vk::ImageView::null() {
                self.device.destroy_image_view(self.source_image_view, None);
            }
            if self.source_image != vk::Image::null() {
                self.device.destroy_image(self.source_image, None);
            }
            if self.source_image_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.source_image_memory, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
