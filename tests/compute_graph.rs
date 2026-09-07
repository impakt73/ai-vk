use std::{
    collections::BTreeMap,
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
fn resolves_graph_arguments_in_dimensions_and_dispatches() {
    let mut overrides = BTreeMap::new();
    overrides.insert("width".into(), "16".into());
    let graph = ComputeGraph::from_toml_with_arguments(
        r#"
            [arguments]
            width = "8"
            height = "4"
            buffer_size = "64"
            groups = "2"

            [resources.image]
            type = "image"
            extent = ["$width", "${height}"]

            [resources.buffer]
            type = "buffer"
            size = "$buffer_size"

            [[nodes]]
            name = "fill"
            shader = "fill.hlsl"
            kernel = "main"
            dispatch = ["$groups", 1, 1]
        "#,
        &overrides,
    )
    .expect("graph arguments should resolve");

    assert_eq!(graph.definition().arguments["width"], "16");
    assert_eq!(graph.definition().arguments["height"], "4");
    assert_eq!(graph.definition().resources["image"].extent, Some([16, 4]));
    assert_eq!(graph.definition().resources["buffer"].size, Some(64));
    assert_eq!(graph.definition().nodes[0].dispatch, [2, 1, 1]);
}

#[test]
fn rejects_undeclared_graph_argument_overrides() {
    let mut overrides = BTreeMap::new();
    overrides.insert("width".into(), "16".into());
    let error = ComputeGraph::from_toml_with_arguments("[arguments]\nheight = \"8\"", &overrides)
        .expect_err("undeclared arguments should be rejected");
    assert!(error.to_string().contains("has no declaration"));
}

#[test]
fn derives_input_resource_dimensions_and_sizes() {
    let image_path = temporary_output("png");
    let buffer_path = temporary_output("bin");
    image::RgbaImage::from_pixel(3, 2, image::Rgba([10, 20, 30, 255]))
        .save(&image_path)
        .expect("input image should be writable");
    fs::write(&buffer_path, [1_u8, 2, 3, 4, 5]).expect("input buffer should be writable");

    let graph = ComputeGraph::from_toml(&format!(
        r#"
            [resources.image]
            type = "image"
            input = "{}"
            format = "rgba8"

            [resources.buffer]
            type = "buffer"
            input = "{}"
        "#,
        image_path.display(),
        buffer_path.display()
    ))
    .expect("input resources should parse");

    assert_eq!(
        graph.definition().resources["image"].input,
        Some(image_path.clone())
    );
    assert_eq!(
        graph.definition().resources["buffer"].input,
        Some(buffer_path.clone())
    );
    fs::remove_file(image_path).expect("input image should be removable");
    fs::remove_file(buffer_path).expect("input buffer should be removable");
}

#[test]
fn rejects_explicit_dimensions_and_missing_image_format_for_inputs() {
    let image_path = temporary_output("png");
    image::RgbaImage::from_pixel(2, 2, image::Rgba([0, 0, 0, 255]))
        .save(&image_path)
        .expect("input image should be writable");

    let explicit_dimensions = ComputeGraph::from_toml(&format!(
        r#"
            [resources.image]
            type = "image"
            input = "{}"
            format = "rgba8"
            extent = [2, 2]
        "#,
        image_path.display()
    ))
    .expect_err("input image dimensions should be derived");
    assert!(
        explicit_dimensions
            .to_string()
            .contains("must not specify dimensions")
    );

    let missing_format = ComputeGraph::from_toml(&format!(
        r#"
            [resources.image]
            type = "image"
            input = "{}"
        "#,
        image_path.display()
    ))
    .expect_err("input image format should be explicit");
    assert!(missing_format.to_string().contains("must specify a format"));
    fs::remove_file(image_path).expect("input image should be removable");
}

#[test]
fn uploads_an_input_image_before_graph_execution() {
    let input_path = temporary_output("png");
    image::RgbaImage::from_fn(8, 8, |x, y| image::Rgba([x as u8, y as u8, 33, 255]))
        .save(&input_path)
        .expect("input image should be writable");

    let graph = ComputeGraph::from_toml(&format!(
        r#"
            [resources.source]
            type = "image"
            input = "{}"
            format = "rgba8"

            [resources.output]
            type = "image"
            extent = [8, 8]

            [[nodes]]
            name = "copy"
            shader = "tests/shaders/graph_copy.hlsl"
            kernel = "main"
            dispatch = [1, 1, 1]
            bindings = [
                {{ resource = "source", access = "read" }},
                {{ resource = "output", access = "write" }},
            ]
        "#,
        input_path.display()
    ))
    .expect("input graph should parse");
    let mut execution = graph
        .execute_with_validation_layers(true)
        .expect("input graph should execute");
    let (width, height, pixels) = execution
        .read_image_rgba8("output")
        .expect("output image should be readable");
    assert_eq!((width, height), (8, 8));
    assert_eq!(&pixels[..4], &[0, 0, 33, 255]);
    assert_eq!(&pixels[(7 * 8 + 3) * 4..(7 * 8 + 4) * 4], &[3, 7, 33, 255]);
    fs::remove_file(input_path).expect("input image should be removable");
}

#[test]
fn uploads_an_input_buffer_before_graph_execution() {
    let input_path = temporary_output("bin");
    let expected = vec![0, 1, 2, 3, 4, 5, 6, 7];
    fs::write(&input_path, &expected).expect("input buffer should be writable");
    let graph = ComputeGraph::from_toml(&format!(
        r#"
            [resources.input]
            type = "buffer"
            input = "{}"
        "#,
        input_path.display()
    ))
    .expect("input buffer graph should parse");
    let mut execution = graph
        .execute_with_validation_layers(true)
        .expect("input buffer graph should execute");
    assert_eq!(
        execution
            .read_buffer("input")
            .expect("buffer should be readable"),
        expected
    );
    fs::remove_file(input_path).expect("input buffer should be removable");
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
            shader = "tests/shaders/graph_fill.hlsl"
            kernel = "main"
            dispatch = [1, 1, 1]
            bindings = [{ resource = "source", access = "write" }]

            [[nodes]]
            name = "copy"
            shader = "tests/shaders/graph_copy.hlsl"
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
            shader = "tests/shaders/graph_buffer_fill.hlsl"
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
            shader = "tests/shaders/graph_fill.hlsl"
            kernel = "main"
            dispatch = [1, 1, 1]
            bindings = [{{ resource = "source", access = "write" }}]

            [[nodes]]
            name = "copy"
            shader = "tests/shaders/graph_copy.hlsl"
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
