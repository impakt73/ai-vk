use std::{
    ffi::{CStr, CString},
    ops::Deref,
    path::Path,
};

use ash::{Entry, vk};

mod compute_graph;

const VALIDATION_LAYER_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

pub(crate) struct VulkanInstance {
    _entry: Entry,
    pub(crate) instance: ash::Instance,
    debug_utils: Option<ash::ext::debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl Deref for VulkanInstance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            if let (Some(debug_utils), Some(debug_messenger)) =
                (&self.debug_utils, self.debug_messenger)
            {
                debug_utils.destroy_debug_utils_messenger(debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

pub(crate) fn create_vulkan_instance(
    entry: Entry,
    app_name: &CStr,
    api_version: u32,
    enable_validation_layers: bool,
) -> Result<VulkanInstance, Box<dyn std::error::Error>> {
    let validation_enabled = if enable_validation_layers {
        let layer_available = unsafe { entry.enumerate_instance_layer_properties()? }
            .iter()
            .any(|layer| unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) } == VALIDATION_LAYER_NAME);
        if !layer_available {
            return Err(format!(
                "requested Vulkan validation layer is not available: {}",
                VALIDATION_LAYER_NAME.to_string_lossy()
            )
            .into());
        }
        true
    } else {
        false
    };
    let debug_utils_available = validation_enabled
        && unsafe { entry.enumerate_instance_extension_properties(None)? }
            .iter()
            .any(|extension| unsafe {
                CStr::from_ptr(extension.extension_name.as_ptr()) == ash::ext::debug_utils::NAME
            });

    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(app_name)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(api_version);
    let layer_names = [VALIDATION_LAYER_NAME.as_ptr()];
    let extension_names = [ash::ext::debug_utils::NAME.as_ptr()];
    let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));
    let mut instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    if validation_enabled {
        instance_info = instance_info.enabled_layer_names(&layer_names);
    }
    if debug_utils_available {
        instance_info = instance_info.enabled_extension_names(&extension_names);
        instance_info = instance_info.push_next(&mut debug_create_info);
    }
    let instance = unsafe { entry.create_instance(&instance_info, None)? };
    let (debug_utils, debug_messenger) = if debug_utils_available {
        let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
        let debug_messenger =
            match unsafe { debug_utils.create_debug_utils_messenger(&debug_create_info, None) } {
                Ok(debug_messenger) => debug_messenger,
                Err(error) => {
                    unsafe { instance.destroy_instance(None) };
                    return Err(error.into());
                }
            };
        (Some(debug_utils), Some(debug_messenger))
    } else {
        (None, None)
    };

    Ok(VulkanInstance {
        _entry: entry,
        instance,
        debug_utils,
        debug_messenger,
    })
}

unsafe extern "system" fn vulkan_debug_callback(
    _message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if !callback_data.is_null() && unsafe { !(*callback_data).p_message.is_null() } {
        let message = unsafe { CStr::from_ptr((*callback_data).p_message) };
        eprintln!("[Vulkan validation] {}", message.to_string_lossy());
    }
    vk::FALSE
}

pub use compute_graph::{
    AccessType, ComputeGraph, ComputeGraphDefinition, ComputeGraphError, ComputeGraphExecution,
    ComputeNodeDefinition, ImageFormat, ResourceBindingDefinition, ResourceDefinition,
    ResourceKind,
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
    enumerate_physical_devices_with_validation_layers(false)
}

pub fn enumerate_physical_devices_with_validation_layers(
    enable_validation_layers: bool,
) -> Result<Vec<PhysicalDeviceInfo>, Box<dyn std::error::Error>> {
    let entry = unsafe { Entry::load()? };
    let app_name = CString::new("ai-vk")?;
    let instance = create_vulkan_instance(
        entry,
        &app_name,
        vk::API_VERSION_1_0,
        enable_validation_layers,
    )?;

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

    Ok(devices)
}

pub fn write_cleared_image_png(
    width: u32,
    height: u32,
    color: [u8; 4],
    output_path: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    write_cleared_image_png_with_validation_layers(width, height, color, output_path, false)
}

pub fn write_cleared_image_png_with_validation_layers(
    width: u32,
    height: u32,
    color: [u8; 4],
    output_path: impl AsRef<Path>,
    enable_validation_layers: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    ComputeGraph::write_solid_color_png(
        width,
        height,
        color,
        output_path,
        enable_validation_layers,
    )?;
    Ok(())
}

pub(crate) fn select_compute_queue(
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

pub(crate) fn find_memory_type(
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

pub(crate) fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}
