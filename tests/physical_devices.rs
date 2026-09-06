use ai_vk::enumerate_physical_devices;

#[test]
fn identifies_a_valid_physical_device_on_the_host() {
    let devices = enumerate_physical_devices().expect("Vulkan device enumeration should succeed");

    assert!(
        devices.iter().any(|device| {
            !device.name.trim().is_empty()
                && device.api_version.0 >= 1
                && device.queue_family_count > 0
        }),
        "expected at least one valid Vulkan physical device, got {devices:?}"
    );
}
