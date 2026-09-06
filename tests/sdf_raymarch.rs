use ai_vk::ComputeGraph;

const WIDTH: u32 = 48;
const HEIGHT: u32 = 32;

#[test]
fn raymarches_three_phong_shaded_spheres() {
    let graph = ComputeGraph::from_toml(&format!(
        r#"
            [resources.output]
            type = "image"
            extent = [{WIDTH}, {HEIGHT}]

            [[nodes]]
            name = "raymarch"
            shader = "shaders/sdf_raymarch.hlsl"
            kernel = "main"
            dispatch = [6, 4, 1]
            bindings = [{{ resource = "output", access = "write" }}]
        "#
    ))
    .expect("SDF raymarch graph should parse");
    let mut execution = graph
        .execute_with_validation_layers(true)
        .expect("SDF raymarch graph should execute");
    let (width, height, pixels) = execution
        .read_image_rgba8("output")
        .expect("SDF raymarch output should be readable");
    let rendered = image::RgbaImage::from_raw(width, height, pixels)
        .expect("SDF raymarch output should have the requested dimensions");
    let reference = image::load_from_memory(include_bytes!("gold/sdf_raymarch.png"))
        .expect("the SDF raymarch gold image should be a valid PNG")
        .into_rgba8();
    let comparison = image_compare::rgba_hybrid_compare(&rendered, &reference)
        .expect("rendered and reference images should have matching dimensions");

    assert!(
        comparison.score >= 0.99,
        "raymarched image differs from the gold image: score {}",
        comparison.score
    );
}
