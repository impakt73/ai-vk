use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use ai_vk::write_bindless_image_png;
use hassle_rs::compile_hlsl;

#[test]
fn compute_shader_writes_a_color_through_the_bindless_image_table() {
    let shader = include_str!("../shaders/solid_color.hlsl");
    let include_path = format!("-I{}/shaders", env!("CARGO_MANIFEST_DIR"));
    let spirv = compile_hlsl(
        "shaders/solid_color.hlsl",
        shader,
        "main",
        "cs_6_0",
        &["-spirv", &include_path],
        &[],
    )
    .expect("bindless HLSL compute shader should compile");
    let output_path = std::env::temp_dir().join(format!(
        "ai-vk-bindless-image-{}-{}.png",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after the Unix epoch")
            .as_nanos()
    ));
    let width = 13;
    let height = 9;
    let color = [12, 98, 201, 255];

    write_bindless_image_png(width, height, color, &spirv, &output_path)
        .expect("bindless compute image rendering should succeed");

    let rendered = image::open(&output_path)
        .expect("the compute output should be a valid PNG")
        .into_rgba8();
    assert_eq!(rendered.dimensions(), (width, height));
    assert!(rendered.pixels().all(|pixel| pixel.0 == color));

    fs::remove_file(output_path).expect("test output should be removable");
}
