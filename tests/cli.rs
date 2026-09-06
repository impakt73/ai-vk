use std::process::Command;

#[test]
fn run_graph_executes_the_checked_in_toml_example() {
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
}
