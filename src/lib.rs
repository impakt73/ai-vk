use std::ffi::{CStr, CString};

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
