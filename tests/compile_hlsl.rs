use hassle_rs::compile_hlsl;

#[test]
fn compiles_an_hlsl_compute_shader_at_runtime() {
    let shader = r#"
        RWStructuredBuffer<uint> output_buffer : register(u0);

        [numthreads(1, 1, 1)]
        void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
        {
            output_buffer[dispatch_thread_id.x] = dispatch_thread_id.x;
        }
    "#;

    let spirv = compile_hlsl(
        "runtime_compute.hlsl",
        shader,
        "main",
        "cs_6_0",
        &["-spirv"],
        &[],
    )
    .expect("HLSL compute shader should compile at runtime");

    assert!(spirv.len() >= 4, "compiled SPIR-V should have a header");
    assert_eq!(
        u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]),
        0x0723_0203,
        "runtime compilation should produce SPIR-V"
    );
}
