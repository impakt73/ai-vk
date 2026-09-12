use std::{fs, path::PathBuf};

use ai_vk::{ComputeGraph, ComputeGraphExecution};

const WIDTH: u32 = 48;
const HEIGHT: u32 = 32;

const OUTPUT_IMAGES: [&str; 4] = [
    "sdf_bilinear_upsampled.png",
    "sdf_bilinear_upsampled_normal.png",
    "sdf_bilinear_upsampled_motion.png",
    "sdf_bilinear_upsampled_depth.png",
];

fn example_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/upsampling")
}

fn assert_matches_gold(
    execution: &mut ComputeGraphExecution,
    resource: &str,
    gold: &[u8],
    label: &str,
) {
    let (width, height, pixels) = execution
        .read_image_rgba8(resource)
        .unwrap_or_else(|error| panic!("example output `{resource}` should be readable: {error}"));
    assert_eq!(
        (width, height),
        (WIDTH, HEIGHT),
        "example output `{resource}` should match the declared output extent"
    );
    let rendered = image::RgbaImage::from_raw(width, height, pixels).unwrap_or_else(|| {
        panic!("example output `{resource}` should have the requested dimensions")
    });
    let reference = image::load_from_memory(gold)
        .unwrap_or_else(|error| panic!("the {label} gold image should be a valid PNG: {error}"))
        .into_rgba8();
    let comparison =
        image_compare::rgba_hybrid_compare(&rendered, &reference).unwrap_or_else(|error| {
            panic!("{label} output and gold image should have matching dimensions: {error}")
        });

    assert!(
        comparison.score >= 0.99,
        "{label} output differs from the gold image: score {}",
        comparison.score
    );
}

#[test]
fn upsampling_example_matches_gold_images() {
    let example = example_directory();
    for output in OUTPUT_IMAGES {
        let _ = fs::remove_file(example.join(output));
    }

    let graph = ComputeGraph::from_toml_file(example.join("sdf_bilinear_upsample.toml"))
        .expect("the upsampling example graph should parse");
    let mut execution = graph
        .execute_with_validation_layers(true)
        .expect("the upsampling example graph should execute");

    assert_matches_gold(
        &mut execution,
        "output",
        include_bytes!("gold/sdf_bilinear_upsampled.png"),
        "color",
    );
    assert_matches_gold(
        &mut execution,
        "output_normal",
        include_bytes!("gold/sdf_bilinear_upsampled_normal.png"),
        "normal",
    );
    assert_matches_gold(
        &mut execution,
        "output_motion",
        include_bytes!("gold/sdf_bilinear_upsampled_motion.png"),
        "motion",
    );
    assert_matches_gold(
        &mut execution,
        "output_depth",
        include_bytes!("gold/sdf_bilinear_upsampled_depth.png"),
        "depth",
    );

    for output in OUTPUT_IMAGES {
        let path = example.join(output);
        let image = image::open(&path)
            .unwrap_or_else(|error| panic!("example should write `{output}`: {error}"));
        assert_eq!((image.width(), image.height()), (WIDTH, HEIGHT));
        fs::remove_file(&path).expect("example output should be removable");
    }
}
