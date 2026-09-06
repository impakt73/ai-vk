use ai_vk::enumerate_physical_devices;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let physical_devices = enumerate_physical_devices()?;

    println!("Vulkan physical devices: {}", physical_devices.len());
    for (index, physical_device) in physical_devices.iter().enumerate() {
        println!("Device {index}: {}", physical_device.name);
        println!("  type: {}", physical_device.device_type);
        println!("  vendor ID: 0x{:04x}", physical_device.vendor_id);
        println!("  device ID: 0x{:04x}", physical_device.device_id);
        println!(
            "  Vulkan API: {}.{}.{}",
            physical_device.api_version.0,
            physical_device.api_version.1,
            physical_device.api_version.2
        );
        println!("  driver version: {}", physical_device.driver_version);
        println!("  queue families: {}", physical_device.queue_family_count);
    }

    Ok(())
}
