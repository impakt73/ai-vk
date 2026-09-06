use std::{fs, process::Command};

use image::GenericImageView;

#[test]
fn run_graph_executes_the_checked_in_toml_example() {
    let output_path = "examples/compute_graph_output.png";
    let _ = fs::remove_file(output_path);
    let output = Command::new(env!("CARGO_BIN_EXE_ai-vk"))
        .args(["run-graph", "examples/compute_graph.toml"])
        .output()
        .expect("CLI should start");

    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("executed compute graph"));
    let image = image::open(output_path).expect("graph should write its declared image output");
    assert_eq!(image.dimensions(), (8, 8));
    fs::remove_file(output_path).expect("test output should be removable");
}
