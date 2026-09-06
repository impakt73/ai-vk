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

## Compute Graphs

`ComputeGraph::from_toml` loads a graph whose resources are allocated once and
kept in persistent bindless slots for the lifetime of the execution. Shader
paths in `ComputeGraph::from_toml_file` are relative to the TOML file.

The CLI can execute an externally authored graph without any Rust code changes:

```text
cargo run -- run-graph examples/compute_graph.toml
```

Use `examples/compute_graph.toml` as a starting point for changing resources,
dispatches, dependencies, and shader paths. The command reports graph parsing,
shader compilation, Vulkan setup, and execution errors directly.

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
