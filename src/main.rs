use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

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
        /// Override a declared graph argument using NAME=VALUE.
        #[arg(
            long = "arg",
            alias = "argument",
            alias = "build-arg",
            value_name = "NAME=VALUE"
        )]
        arguments: Vec<String>,
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
        Some(Command::RunGraph { graph, arguments }) => {
            run_graph(&graph, cli.validation_layers, &arguments)
        }
        None => list_physical_devices(cli.validation_layers),
    }
}

fn run_graph(
    path: &Path,
    enable_validation_layers: bool,
    raw_arguments: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let arguments = parse_arguments(raw_arguments)?;
    let graph = ComputeGraph::from_toml_file_with_arguments(path, &arguments)?;
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

fn parse_arguments(raw_arguments: &[String]) -> Result<BTreeMap<String, String>, String> {
    let mut arguments = BTreeMap::new();
    for raw in raw_arguments {
        let Some((name, value)) = raw.split_once('=') else {
            return Err(format!("graph argument `{raw}` must use NAME=VALUE syntax"));
        };
        if name.is_empty() {
            return Err("graph argument names must not be empty".into());
        }
        if arguments
            .insert(name.to_owned(), value.to_owned())
            .is_some()
        {
            return Err(format!(
                "graph argument `{name}` was specified more than once"
            ));
        }
    }
    Ok(arguments)
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
