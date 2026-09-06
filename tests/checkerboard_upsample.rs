use ai_vk::ComputeGraph;

const SOURCE_SIZE: u32 = 16;
const OUTPUT_SIZE: u32 = 32;

#[test]
fn generates_a_checkerboard_and_bilinearly_upsamples_it() {
    let graph = ComputeGraph::from_toml(
        r#"
            [resources.source]
            type = "image"
            extent = [16, 16]

            [resources.output]
            type = "image"
            extent = [32, 32]

            [[nodes]]
            name = "generate-checkerboard"
            shader = "tests/shaders/checkerboard.hlsl"
            kernel = "main"
            dispatch = [2, 2, 1]
            bindings = [{ resource = "source", access = "write" }]

            [[nodes]]
            name = "bilinear-upsample"
            shader = "tests/shaders/bilinear_upsample.hlsl"
            kernel = "main"
            dispatch = [4, 4, 1]
            bindings = [
                { resource = "source", access = "read" },
                { resource = "output", access = "write" },
            ]
        "#,
    )
    .expect("checkerboard graph should parse");
    let mut execution = graph
        .execute_with_validation_layers(true)
        .expect("checkerboard graph should execute");

    let (source_width, source_height, source_pixels) = execution
        .read_image_rgba8("source")
        .expect("checkerboard source should be readable");
    assert_eq!((source_width, source_height), (SOURCE_SIZE, SOURCE_SIZE));
    for y in 0..SOURCE_SIZE {
        for x in 0..SOURCE_SIZE {
            let pixel = &source_pixels[((y * SOURCE_SIZE + x) * 4) as usize..];
            let expected = if (x / 4 + y / 4) % 2 == 0 {
                [0, 0, 0, 255]
            } else {
                [255, 255, 255, 255]
            };
            assert_eq!(
                &pixel[..4],
                &expected,
                "unexpected source pixel at ({x}, {y})"
            );
        }
    }

    let (width, height, pixels) = execution
        .read_image_rgba8("output")
        .expect("upsampled output should be readable");
    assert_eq!((width, height), (OUTPUT_SIZE, OUTPUT_SIZE));
    assert_eq!(&pixels[..4], &[0, 0, 0, 255]);
    assert_eq!(
        &pixels[((0 * OUTPUT_SIZE + 7) * 4) as usize..][..4],
        &[64, 64, 64, 255]
    );
    assert_eq!(
        &pixels[((7 * OUTPUT_SIZE + 7) * 4) as usize..][..4],
        &[96, 96, 96, 255]
    );
}
