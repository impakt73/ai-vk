use std::path::{Path, PathBuf};

use ai_vk::{ComputeGraph, enumerate_physical_devices_with_validation_layers};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "ai-vk", version, about = "Vulkan compute image utilities")]
struct Cli {
    /// Enable the Vulkan Khronos validation layers.
    #[arg(long)]
    validation_layers: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Load, compile, and execute a compute graph from a TOML file.
    RunGraph {
        /// Compute graph TOML file.
        graph: PathBuf,
    },
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::RunGraph { graph }) => run_graph(&graph, cli.validation_layers),
        None => list_physical_devices(cli.validation_layers),
    }
}

fn run_graph(
    path: &Path,
    enable_validation_layers: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let graph = ComputeGraph::from_toml_file(path)?;
    let node_count = graph.definition().nodes.len();
    let resource_count = graph.definition().resources.len();
    graph.execute_with_validation_layers(enable_validation_layers)?;
    println!(
        "executed compute graph {} ({} nodes, {} resources)",
        path.display(),
        node_count,
        resource_count
    );
    Ok(())
}

fn list_physical_devices(enable_validation_layers: bool) -> Result<(), Box<dyn std::error::Error>> {
    let physical_devices =
        enumerate_physical_devices_with_validation_layers(enable_validation_layers)?;

    println!("Vulkan physical devices: {}", physical_devices.len());
    for (index, physical_device) in physical_devices.iter().enumerate() {
        println!("Device {index}: {}", physical_device.name);
        println!("  type: {}", physical_device.device_type);
        println!("  vendor ID: 0x{:04x}", physical_device.vendor_id);
        println!("  device ID: 0x{:04x}", physical_device.device_id);
        println!(
            "  Vulkan API: {}.{}.{}",
            physical_device.api_version.0,
            physical_device.api_version.1,
            physical_device.api_version.2
        );
        println!("  driver version: {}", physical_device.driver_version);
        println!("  queue families: {}", physical_device.queue_family_count);
    }

    Ok(())
}
