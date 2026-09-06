use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use ai_vk::ComputeGraph;

#[test]
fn clears_an_image_on_a_compute_queue_and_writes_a_png() {
    let output_path = std::env::temp_dir().join(format!(
        "ai-vk-compute-image-{}-{}.png",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after the Unix epoch")
            .as_nanos()
    ));
    let width = 7;
    let height = 5;
    let color = [12, 98, 201, 255];

    let graph = ComputeGraph::from_toml(&format!(
        r#"
            [resources.output]
            type = "image"
            extent = [{width}, {height}]
            output = "{}"

            [[nodes]]
            name = "fill"
            shader = "shaders/graph_fill.hlsl"
            kernel = "main"
            dispatch = [1, 1, 1]
            bindings = [{{ resource = "output", access = "write" }}]
        "#,
        output_path.display()
    ))
    .expect("compute image graph should parse");
    graph
        .execute_with_validation_layers(true)
        .expect("compute image graph should execute");

    let rendered = image::open(&output_path)
        .expect("the graph output should be a valid PNG")
        .into_rgba8();
    assert_eq!(rendered.dimensions(), (width, height));
    assert!(rendered.pixels().all(|pixel| pixel.0 == color));

    fs::remove_file(output_path).expect("test output should be removable");
}
