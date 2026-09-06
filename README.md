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
