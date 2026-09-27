use std::{
    fs,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

use image::GenericImageView;

#[test]
fn run_graph_executes_the_checked_in_toml_example() {
    let output_path = "examples/compute_graph_output.png";
    let _ = fs::remove_file(output_path);
    let output = Command::new(env!("CARGO_BIN_EXE_ai-vk"))
        .args([
            "--validation-layers",
            "run-graph",
            "examples/compute_graph.toml",
            "--arg",
            "width=4",
            "--arg",
            "height=4",
        ])
        .output()
        .expect("CLI should start");

    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("executed compute graph"));
    assert!(String::from_utf8_lossy(&output.stdout).contains("GPU execution time:"));
    let image = image::open(output_path).expect("graph should write its declared image output");
    assert_eq!(image.dimensions(), (4, 4));
    fs::remove_file(output_path).expect("test output should be removable");
}

#[test]
fn run_graph_executes_temporal_frames_and_writes_separate_outputs() {
    static NEXT_OUTPUT: AtomicUsize = AtomicUsize::new(0);
    let index = NEXT_OUTPUT.fetch_add(1, Ordering::Relaxed);
    let output_directory =
        std::env::temp_dir().join(format!("ai-vk-cli-temporal-{}-{index}", std::process::id()));
    let output_pattern = output_directory.join("frame-{frame}.png");
    let output = Command::new(env!("CARGO_BIN_EXE_ai-vk"))
        .args([
            "--validation-layers",
            "run-graph",
            "examples/temporal/temporal_accumulation.toml",
            "--frames",
            "2",
            "--arg",
        ])
        .arg(format!("output_pattern={}", output_pattern.display()))
        .output()
        .expect("CLI should start");

    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("2 frames"));
    assert!(stdout.contains("frame 0 GPU execution time:"));
    assert!(stdout.contains("frame 1 GPU execution time:"));
    for (frame, expected_channel) in [63, 127].into_iter().enumerate() {
        let path = output_directory.join(format!("frame-{frame}.png"));
        let image = image::open(&path)
            .unwrap_or_else(|error| panic!("frame {frame} output should be written: {error}"))
            .into_rgba8();
        assert!(image.pixels().all(|pixel| {
            pixel.0 == [expected_channel, expected_channel, expected_channel, 255]
        }));
        fs::remove_file(path).expect("frame output should be removable");
    }
    fs::remove_dir(output_directory).expect("output directory should be removable");
}
