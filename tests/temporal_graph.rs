use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use ai_vk::ComputeGraph;

fn temporary_output_directory() -> PathBuf {
    static NEXT_OUTPUT: AtomicUsize = AtomicUsize::new(0);
    let index = NEXT_OUTPUT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "ai-vk-temporal-graph-{}-{index}",
        std::process::id()
    ))
}

#[test]
fn executes_temporal_accumulation_and_dumps_each_frame() {
    let output_directory = temporary_output_directory();
    let output_pattern = output_directory.join("accumulation-{frame}.png");
    let mut arguments = BTreeMap::new();
    arguments.insert(
        "output_pattern".to_owned(),
        output_pattern.to_string_lossy().into_owned(),
    );
    let graph = ComputeGraph::from_toml_file_with_arguments(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples/temporal/temporal_accumulation.toml"),
        &arguments,
    )
    .expect("temporal accumulation graph should parse");

    let mut execution = graph
        .execute_frames_with_validation_layers(4, true)
        .expect("temporal accumulation graph should execute four frames");
    assert_eq!(execution.frame_gpu_execution_times().len(), 4);

    for (frame, expected_channel) in [63, 127, 191, 255].into_iter().enumerate() {
        let path = output_directory.join(format!("accumulation-{frame}.png"));
        let image = image::open(&path)
            .unwrap_or_else(|error| panic!("frame {frame} output should be written: {error}"))
            .into_rgba8();
        assert_eq!(image.dimensions(), (8, 8));
        assert!(
            image
                .pixels()
                .all(|pixel| pixel.0 == [expected_channel, expected_channel, expected_channel, 255]),
            "frame {frame} should contain accumulated value {expected_channel}"
        );
        fs::remove_file(path).expect("frame output should be removable");
    }
    fs::remove_dir(output_directory).expect("output directory should be removable");

    let (width, height, pixels) = execution
        .read_image_rgba8("accumulation")
        .expect("final temporal resource should be readable");
    assert_eq!((width, height), (8, 8));
    assert!(
        pixels
            .as_chunks::<4>()
            .0
            .iter()
            .all(|pixel| *pixel == [255, 255, 255, 255]),
        "the resource should contain the four-frame accumulated result"
    );
}
