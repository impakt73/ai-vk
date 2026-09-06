use ai_vk::ComputeGraph;

const WIDTH: u32 = 48;
const HEIGHT: u32 = 32;

#[test]
fn raymarches_and_bilinearly_downsamples_three_phong_shaded_spheres() {
    let graph = ComputeGraph::from_toml(&format!(
        r#"
            [resources.source]
            type = "image"
            extent = [{WIDTH}, {HEIGHT}]

            [resources.output]
            type = "image"
            extent = [{output_width}, {output_height}]

            [[nodes]]
            name = "raymarch"
            shader = "tests/shaders/sdf_raymarch.hlsl"
            kernel = "main"
            dispatch = [6, 4, 1]
            bindings = [{{ resource = "source", access = "write" }}]

            [[nodes]]
            name = "downsample"
            shader = "tests/shaders/sdf_downsample.hlsl"
            kernel = "main"
            dispatch = [3, 2, 1]
            bindings = [
                {{ resource = "source", access = "read" }},
                {{ resource = "output", access = "write" }},
            ]
        "#,
        output_width = WIDTH / 2,
        output_height = HEIGHT / 2,
    ))
    .expect("SDF downsample graph should parse");
    let mut execution = graph
        .execute_with_validation_layers(true)
        .expect("SDF downsample graph should execute");
    let (width, height, pixels) = execution
        .read_image_rgba8("output")
        .expect("SDF downsample output should be readable");
    let rendered = image::RgbaImage::from_raw(width, height, pixels)
        .expect("SDF downsample output should have the requested dimensions");
    let reference = image::load_from_memory(include_bytes!("gold/sdf_raymarch_downsample.png"))
        .expect("the SDF raymarch downsample gold image should be a valid PNG")
        .into_rgba8();
    let comparison = image_compare::rgba_hybrid_compare(&rendered, &reference)
        .expect("rendered and reference images should have matching dimensions");

    assert_eq!(rendered.dimensions(), (WIDTH / 2, HEIGHT / 2));
    assert!(
        comparison.score >= 0.99,
        "downsampled raymarched image differs from the gold image: score {}",
        comparison.score
    );
}
