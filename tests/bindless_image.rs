use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use ai_vk::ComputeGraph;

#[test]
fn compute_shader_writes_a_color_through_the_bindless_image_table() {
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

    let graph = ComputeGraph::from_toml(&format!(
        r#"
            [resources.output]
            type = "image"
            extent = [{width}, {height}]
            output = "{}"

            [[nodes]]
            name = "fill"
            shader = "tests/shaders/solid_color.hlsl"
            kernel = "main"
            dispatch = [2, 2, 1]
            bindings = [{{ resource = "output", access = "write" }}]
        "#,
        output_path.display()
    ))
    .expect("bindless image graph should parse");
    graph
        .execute_with_validation_layers(true)
        .expect("bindless image graph should execute");

    let rendered = image::open(&output_path)
        .expect("the graph output should be a valid PNG")
        .into_rgba8();
    assert_eq!(rendered.dimensions(), (width, height));
    assert!(rendered.pixels().all(|pixel| pixel.0 == color));

    fs::remove_file(output_path).expect("test output should be removable");
}
