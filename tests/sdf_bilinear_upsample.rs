use std::{fs, path::PathBuf};

use ai_vk::ComputeGraph;

const WIDTH: u32 = 48;
const HEIGHT: u32 = 32;
const FRAME_COUNT: usize = 2;

const OUTPUT_IMAGES: [&str; 4] = [
    "sdf_bilinear_upsampled-{frame}.png",
    "sdf_bilinear_upsampled_normal-{frame}.png",
    "sdf_bilinear_upsampled_motion-{frame}.png",
    "sdf_bilinear_upsampled_depth-{frame}.png",
];
const TEMPORAL_VALIDITY_OUTPUT: &str = "sdf_bilinear_temporal_validity-{frame}.png";

fn example_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/upsampling")
}

fn assert_path_matches_gold(path: &std::path::Path, gold: &[u8], label: &str) {
    let rendered = image::open(path)
        .unwrap_or_else(|error| {
            panic!(
                "example output `{}` should be readable: {error}",
                path.display()
            )
        })
        .into_rgba8();
    assert_eq!(rendered.dimensions(), (WIDTH, HEIGHT));
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

fn output_path(example: &std::path::Path, pattern: &str, frame: usize) -> PathBuf {
    example.join(pattern.replace("{frame}", &frame.to_string()))
}

#[test]
fn upsampling_example_matches_gold_images() {
    let example = example_directory();
    let output_patterns = OUTPUT_IMAGES.into_iter().chain([TEMPORAL_VALIDITY_OUTPUT]);
    for output in output_patterns.clone() {
        for frame in 0..FRAME_COUNT {
            let _ = fs::remove_file(output_path(&example, output, frame));
        }
    }

    let graph = ComputeGraph::from_toml_file(example.join("sdf_bilinear_upsample.toml"))
        .expect("the upsampling example graph should parse");
    let mut execution = graph
        .execute_frames_with_validation_layers(FRAME_COUNT, true)
        .expect("the upsampling example graph should execute two frames");

    assert_path_matches_gold(
        &output_path(&example, OUTPUT_IMAGES[0], 0),
        include_bytes!("gold/sdf_bilinear_upsampled.png"),
        "color",
    );
    assert_path_matches_gold(
        &output_path(&example, OUTPUT_IMAGES[1], 0),
        include_bytes!("gold/sdf_bilinear_upsampled_normal.png"),
        "normal",
    );
    assert_path_matches_gold(
        &output_path(&example, OUTPUT_IMAGES[2], 0),
        include_bytes!("gold/sdf_bilinear_upsampled_motion.png"),
        "motion",
    );
    assert_path_matches_gold(
        &output_path(&example, OUTPUT_IMAGES[3], 0),
        include_bytes!("gold/sdf_bilinear_upsampled_depth.png"),
        "depth",
    );

    let first_frame_validity = image::open(output_path(&example, TEMPORAL_VALIDITY_OUTPUT, 0))
        .expect("the first-frame temporal validity image should be written")
        .into_rgba8();
    assert_eq!(first_frame_validity.dimensions(), (WIDTH, HEIGHT));
    assert!(
        first_frame_validity
            .pixels()
            .all(|pixel| pixel.0 == [255, 0, 0, 255]),
        "frame zero has no history and should reject every pixel"
    );

    let second_frame_validity = image::open(output_path(&example, TEMPORAL_VALIDITY_OUTPUT, 1))
        .expect("the second-frame temporal validity image should be written")
        .into_rgba8();
    assert_eq!(second_frame_validity.dimensions(), (WIDTH, HEIGHT));
    assert_path_matches_gold(
        &output_path(&example, TEMPORAL_VALIDITY_OUTPUT, 1),
        include_bytes!("gold/sdf_bilinear_temporal_validity.png"),
        "temporal validity",
    );
    let mut accepted = 0;
    let mut rejected = 0;
    for pixel in second_frame_validity.pixels() {
        match pixel.0 {
            [0, 255, 0, 255] => accepted += 1,
            [255, 0, 0, 255] => rejected += 1,
            other => panic!("temporal validity pixels should be binary red/green: {other:?}"),
        }
    }
    assert!(
        accepted > 0,
        "the second frame should contain accepted pixels"
    );
    assert!(
        rejected > 0,
        "the second frame should contain rejected pixels"
    );

    for output in output_patterns {
        for frame in 0..FRAME_COUNT {
            fs::remove_file(output_path(&example, output, frame))
                .expect("example output should be removable");
        }
    }

    let (width, height, pixels) = execution
        .read_image_rgba8("temporal_validity")
        .expect("the final temporal validity image should be readable");
    assert_eq!((width, height), (WIDTH, HEIGHT));
    assert_eq!(pixels, second_frame_validity.into_raw());
}
