use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use ai_vk::{AccessType, ComputeGraph, ResourceKind};
use image::GenericImageView;

fn temporary_output(extension: &str) -> PathBuf {
    static NEXT_OUTPUT: AtomicUsize = AtomicUsize::new(0);
    let index = NEXT_OUTPUT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "ai-vk-compute-graph-{}-{index}.{extension}",
        std::process::id()
    ))
}

#[test]
fn parses_resources_nodes_and_resource_hazards_from_toml() {
    let graph = ComputeGraph::from_toml(
        r#"
            [resources.source]
            type = "image"
            width = 8
            height = 8

            [resources.output]
            type = "buffer"
            size = 256

            [[nodes]]
            name = "produce"
            shader = "produce.hlsl"
            kernel = "main"
            dispatch = [1, 1, 1]
            bindings = [{ resource = "source", access = "write" }]

            [[nodes]]
            name = "consume"
            shader = "consume.hlsl"
            kernel = "main"
            dispatch = [1, 1, 1]
            bindings = [
                { resource = "source", access = "read" },
                { resource = "output", access = "write" },
            ]
        "#,
    )
    .expect("graph TOML should parse");

    assert_eq!(graph.resource_slot("output"), Some(0));
    assert_eq!(graph.resource_slot("source"), Some(1));
    assert_eq!(
        graph.definition().resources["source"].kind,
        ResourceKind::Image
    );
    assert_eq!(
        graph.definition().nodes[0].bindings[0].access,
        AccessType::Write
    );
    assert_eq!(graph.dependencies(0), Some([].as_slice()));
    assert_eq!(graph.dependencies(1), Some([0].as_slice()));
}

#[test]
fn rejects_unknown_resources_and_invalid_dimensions() {
    let unknown = ComputeGraph::from_toml(
        r#"
            [[nodes]]
            name = "bad"
            shader = "bad.hlsl"
            kernel = "main"
            dispatch = [1, 1, 1]
            bindings = [{ resource = "missing", access = "read" }]
        "#,
    )
    .expect_err("unknown resources should be rejected");
    assert!(unknown.to_string().contains("undeclared resource"));

    let invalid_image = ComputeGraph::from_toml(
        r#"
            [resources.output]
            type = "image"
            width = 0
            height = 8
        "#,
    )
    .expect_err("zero-sized images should be rejected");
    assert!(invalid_image.to_string().contains("non-zero width"));

    let invalid_output = ComputeGraph::from_toml(
        r#"
            [resources.output]
            type = "image"
            extent = [8, 8]
            output = "output.raw"
        "#,
    )
    .expect_err("image outputs without a .png extension should be rejected");
    assert!(invalid_output.to_string().contains(".png extension"));
}

#[test]
fn executes_dependent_dispatches_and_retains_bindless_slots() {
    let graph = ComputeGraph::from_toml(
        r#"
            [resources.source]
            type = "image"
            width = 8
            height = 8

            [resources.output]
            type = "image"
            width = 8
            height = 8

            [[nodes]]
            name = "fill"
            shader = "shaders/graph_fill.hlsl"
            kernel = "main"
            dispatch = [1, 1, 1]
            bindings = [{ resource = "source", access = "write" }]

            [[nodes]]
            name = "copy"
            shader = "shaders/graph_copy.hlsl"
            kernel = "main"
            dispatch = [1, 1, 1]
            bindings = [
                { resource = "source", access = "read" },
                { resource = "output", access = "write" },
            ]
        "#,
    )
    .expect("graph TOML should parse");
    let mut execution = graph
        .execute_with_validation_layers(true)
        .expect("compute graph should execute");
    assert_eq!(
        execution.resource_slot("source"),
        graph.resource_slot("source")
    );
    assert_eq!(
        execution.resource_slot("output"),
        graph.resource_slot("output")
    );
    let (width, height, pixels) = execution
        .read_image_rgba8("output")
        .expect("graph output should be readable");
    assert_eq!((width, height), (8, 8));
    assert!(
        pixels
            .as_chunks::<4>()
            .0
            .iter()
            .all(|pixel| *pixel == [12, 98, 201, 255])
    );
}

#[test]
fn creates_a_persistent_buffer_slot_and_executes_a_buffer_dispatch() {
    let output_path = temporary_output("bin");
    let graph = ComputeGraph::from_toml(&format!(
        r#"
            [resources.output]
            type = "buffer"
            size = 64
            output = "{}"

            [[nodes]]
            name = "fill"
            shader = "shaders/graph_buffer_fill.hlsl"
            kernel = "main"
            dispatch = [16, 1, 1]
            bindings = [{{ resource = "output", access = "write" }}]
        "#,
        output_path.display()
    ))
    .expect("graph TOML should parse");
    let mut execution = graph
        .execute_with_validation_layers(true)
        .expect("buffer graph should execute");
    assert_eq!(execution.resource_slot("output"), Some(0));
    let bytes = execution
        .read_buffer("output")
        .expect("graph buffer should be readable");
    let values: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect();
    assert_eq!(
        values,
        (0..16).map(|index| index * 3 + 7).collect::<Vec<_>>()
    );
    let output = fs::read(&output_path).expect("buffer output should be written as raw bytes");
    assert_eq!(output, bytes);
    fs::remove_file(output_path).expect("test output should be removable");
}

#[test]
fn writes_a_declared_image_output_after_graph_execution() {
    let output_path = temporary_output("PNG");
    let graph = ComputeGraph::from_toml(&format!(
        r#"
            [resources.source]
            type = "image"
            extent = [8, 8]

            [resources.output]
            type = "image"
            extent = [8, 8]
            output = "{}"

            [[nodes]]
            name = "fill"
            shader = "shaders/graph_fill.hlsl"
            kernel = "main"
            dispatch = [1, 1, 1]
            bindings = [{{ resource = "source", access = "write" }}]

            [[nodes]]
            name = "copy"
            shader = "shaders/graph_copy.hlsl"
            kernel = "main"
            dispatch = [1, 1, 1]
            bindings = [
                {{ resource = "source", access = "read" }},
                {{ resource = "output", access = "write" }},
            ]
        "#,
        output_path.display()
    ))
    .expect("graph TOML should parse");

    graph
        .execute_with_validation_layers(true)
        .expect("graph should write its image output");
    let image = image::open(&output_path).expect("declared image output should be a PNG");
    assert_eq!(image.dimensions(), (8, 8));
    assert_eq!(image.to_rgba8().get_pixel(0, 0).0, [12, 98, 201, 255]);
    fs::remove_file(output_path).expect("test output should be removable");
}
