use std::{
    ffi::{CStr, CString},
    path::Path,
};

use ash::{Entry, vk};

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
        image: vk::Image::null(),
        image_memory: vk::DeviceMemory::null(),
        buffer: vk::Buffer::null(),
        buffer_memory: vk::DeviceMemory::null(),
        command_pool: vk::CommandPool::null(),
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

struct ComputeResources {
    instance: ash::Instance,
    device: ash::Device,
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    command_pool: vk::CommandPool,
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
            if self.image != vk::Image::null() {
                self.device.destroy_image(self.image, None);
            }
            if self.image_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.image_memory, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
