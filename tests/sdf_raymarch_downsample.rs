use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use ai_vk::write_sdf_downsampled_image_png_with_validation_layers;
use hassle_rs::compile_hlsl;

const WIDTH: u32 = 48;
const HEIGHT: u32 = 32;

#[test]
fn raymarches_and_bilinearly_downsamples_three_phong_shaded_spheres() {
    let include_path = format!("-I{}/shaders", env!("CARGO_MANIFEST_DIR"));
    let sdf_spirv = compile_hlsl(
        "shaders/sdf_raymarch.hlsl",
        include_str!("../shaders/sdf_raymarch.hlsl"),
        "main",
        "cs_6_0",
        &["-spirv", &include_path],
        &[],
    )
    .expect("SDF raymarch HLSL compute shader should compile");
    let downsample_spirv = compile_hlsl(
        "shaders/sdf_downsample.hlsl",
        include_str!("../shaders/sdf_downsample.hlsl"),
        "main",
        "cs_6_0",
        &["-spirv", &include_path],
        &[],
    )
    .expect("SDF downsample HLSL compute shader should compile");

    let output_path = std::env::temp_dir().join(format!(
        "ai-vk-sdf-raymarch-downsample-{}-{}.png",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after the Unix epoch")
            .as_nanos()
    ));
    write_sdf_downsampled_image_png_with_validation_layers(
        WIDTH,
        HEIGHT,
        &sdf_spirv,
        &downsample_spirv,
        &output_path,
        true,
    )
    .expect("SDF raymarch downsampling should succeed");

    let rendered = image::open(&output_path)
        .expect("the downsampled output should be a valid PNG")
        .into_rgba8();
    let reference = image::load_from_memory(include_bytes!("gold/sdf_raymarch_downsample.png"))
        .expect("the SDF raymarch downsample gold image should be a valid PNG")
        .into_rgba8();
    let comparison = image_compare::rgba_hybrid_compare(&rendered, &reference)
        .expect("rendered and reference images should have matching dimensions");

    fs::remove_file(output_path).expect("test output should be removable");
    assert_eq!(rendered.dimensions(), (WIDTH / 2, HEIGHT / 2));
    assert!(
        comparison.score >= 0.99,
        "downsampled raymarched image differs from the gold image: score {}",
        comparison.score
    );
}
