use std::path::{Path, PathBuf};

use ai_vk::{
    ComputeGraph, enumerate_physical_devices_with_validation_layers,
    write_cleared_image_png_with_validation_layers,
};
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
    /// Create a solid-color PNG using a compute queue.
    CreateImage {
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
        /// Color as #RRGGBB, #RRGGBBAA, R,G,B, or R,G,B,A.
        #[arg(value_parser = parse_color)]
        color: [u8; 4],
        /// Destination PNG path.
        output: PathBuf,
    },
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
        Some(Command::CreateImage {
            width,
            height,
            color,
            output,
        }) => {
            write_cleared_image_png_with_validation_layers(
                width,
                height,
                color,
                &output,
                cli.validation_layers,
            )?;
            println!("wrote {width}x{height} image to {}", output.display());
            Ok(())
        }
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
fn parse_color(value: &str) -> Result<[u8; 4], String> {
    let hex = value.strip_prefix('#').unwrap_or(value);
    if matches!(hex.len(), 6 | 8) {
        let mut color = [0, 0, 0, 255];
        for (index, channel) in color.iter_mut().enumerate().take(hex.len() / 2) {
            *channel = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16)
                .map_err(|_| format!("invalid hexadecimal color: {value}"))?;
        }
        return Ok(color);
    }

    let channels: Vec<u8> = value
        .split(',')
        .map(|channel| channel.trim().parse::<u8>())
        .collect::<Result<_, _>>()
        .map_err(|_| format!("invalid color; use #RRGGBB[AA] or R,G,B[,A]: {value}"))?;
    match channels.as_slice() {
        [red, green, blue] => Ok([*red, *green, *blue, 255]),
        [red, green, blue, alpha] => Ok([*red, *green, *blue, *alpha]),
        _ => Err(format!(
            "invalid color; use #RRGGBB[AA] or R,G,B[,A]: {value}"
        )),
    }
}
