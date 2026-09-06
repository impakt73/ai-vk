# ai-vk

## Tests

The HLSL runtime compilation test uses `hassle-rs`, which loads
`libdxcompiler.so` through the system dynamic linker. The Vulkan SDK environment
must provide that library on `LD_LIBRARY_PATH`; no project-local DXC path is
required.

With the Vulkan SDK environment active, run the complete test suite with:

```text
cargo test --all-targets
```

Vulkan integration tests enable `VK_LAYER_KHRONOS_validation` and report
validation errors and warnings through the test output. The layer must be
available through the active Vulkan SDK environment.

## Compute Graphs

`ComputeGraph::from_toml` loads a graph whose resources are allocated once and
kept in persistent bindless slots for the lifetime of the execution. Shader
paths in `ComputeGraph::from_toml_file` are relative to the TOML file.

The CLI can execute an externally authored graph without any Rust code changes:

```text
cargo run -- run-graph examples/compute_graph.toml
```

Validation is optional for the CLI. Add `--validation-layers` before the
subcommand to enable it for image creation, graph execution, or device listing:

```text
cargo run -- --validation-layers run-graph examples/compute_graph.toml
```

Use `examples/compute_graph.toml` as a starting point for changing resources,
dispatches, dependencies, and shader paths. The command reports graph parsing,
shader compilation, Vulkan setup, and execution errors directly.

Resources can declare an output path. Paths are relative to the graph TOML file;
images are written as PNG files and buffers are written as raw binary after all
graph dispatches have completed:

```toml
[resources.image]
type = "image"
extent = [512, 512]
output = "result.png"

[resources.data]
type = "buffer"
size = 1024
output = "result.bin"
```

```toml
[resources.source]
type = "image"
extent = [8, 8]

[resources.output]
type = "buffer"
size = 256

[[nodes]]
name = "produce"
shader = "produce.hlsl"
kernel = "main"
dispatch = [1, 1, 1]
bindings = [{ resource = "source", access = "write" }]
```

Graph shaders should include `shaders/compute_graph.hlsl`. Each dispatch gets a
120-byte push-constant table containing `compute_graph.slots[]`. A slot indexes
`bindless_images[]` or `bindless_buffers[]` depending on the resource type.
Image sampling is available through `bindless_textures[]` from the same include.
