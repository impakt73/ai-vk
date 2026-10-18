use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use ai_vk::ComputeGraph;

fn example_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/rasterization")
}

fn temporary_output_path() -> PathBuf {
    static NEXT_OUTPUT: AtomicUsize = AtomicUsize::new(0);
    let index = NEXT_OUTPUT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "ai-vk-triangle-rasterize-{}-{index}.png",
        std::process::id()
    ))
}

fn is_triangle_color(pixel: &[u8]) -> bool {
    pixel[0] > 240
        && (110..=120).contains(&pixel[1])
        && (20..=30).contains(&pixel[2])
        && pixel[3] == 255
}

fn is_background_color(pixel: &[u8]) -> bool {
    pixel[0] < 20 && pixel[1] < 20 && pixel[2] < 30 && pixel[3] == 255
}

#[test]
fn rasterizes_triangle_buffer_into_the_output_image() {
    let example = example_directory();
    let output_path = example.join("triangle_rasterized.png");
    let _ = fs::remove_file(&output_path);

    let graph = ComputeGraph::from_toml_file(example.join("triangle_rasterize.toml"))
        .expect("the triangle rasterization example graph should parse");
    let mut execution = graph
        .execute_with_validation_layers(true)
        .expect("the triangle rasterization example graph should execute");

    let (width, height, pixels) = execution
        .read_image_rgba8("output")
        .expect("the rasterized image should be readable");
    assert_eq!((width, height), (128, 128));
    assert_eq!(pixels.len(), width as usize * height as usize * 4);

    let pixel = |x: u32, y: u32| {
        let offset = (y as usize * width as usize + x as usize) * 4;
        &pixels[offset..offset + 4]
    };

    assert!(
        is_triangle_color(pixel(width / 2, height / 2)),
        "the image center should be covered by the triangle: {:?}",
        pixel(width / 2, height / 2)
    );
    assert!(
        is_background_color(pixel(0, 0)),
        "the top-left corner should be background: {:?}",
        pixel(0, 0)
    );

    let mut covered = 0;
    for chunk in pixels.as_chunks::<4>().0 {
        if is_triangle_color(chunk) {
            covered += 1;
        } else {
            assert!(
                is_background_color(chunk),
                "pixels should be either triangle color or background: {chunk:?}"
            );
        }
    }
    assert!(covered > 0, "the triangle should cover at least one pixel");
    assert!(
        covered < pixels.len() / 4,
        "the triangle should not cover the whole image"
    );

    assert!(
        output_path.exists(),
        "the example should write its output image"
    );
    fs::remove_file(output_path).expect("test output should be removable");
}

#[test]
fn rasterized_triangle_matches_gold_image() {
    let output_path = temporary_output_path();
    let mut arguments = BTreeMap::new();
    arguments.insert(
        "output".to_owned(),
        output_path.to_string_lossy().into_owned(),
    );

    let graph = ComputeGraph::from_toml_file_with_arguments(
        example_directory().join("triangle_rasterize.toml"),
        &arguments,
    )
    .expect("the triangle rasterization example graph should parse");
    graph
        .execute_with_validation_layers(true)
        .expect("the triangle rasterization example graph should execute");

    let rendered = image::open(&output_path)
        .unwrap_or_else(|error| {
            panic!(
                "example output `{}` should be readable: {error}",
                output_path.display()
            )
        })
        .into_rgba8();
    let reference = image::load_from_memory(include_bytes!("gold/triangle_rasterized.png"))
        .expect("the triangle rasterization gold image should be a valid PNG")
        .into_rgba8();
    assert_eq!(rendered.dimensions(), reference.dimensions());
    let comparison = image_compare::rgba_hybrid_compare(&rendered, &reference)
        .expect("rasterized output and gold image should have matching dimensions");
    assert!(
        comparison.score >= 0.99,
        "rasterized output differs from the gold image: score {}",
        comparison.score
    );

    fs::remove_file(output_path).expect("test output should be removable");
}
