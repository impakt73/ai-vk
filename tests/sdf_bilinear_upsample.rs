use std::{fs, path::PathBuf};

use ai_vk::ComputeGraph;

const WIDTH: u32 = 48;
const HEIGHT: u32 = 32;

const OUTPUT_IMAGES: [&str; 4] = [
    "sdf_bilinear_upsampled.png",
    "sdf_bilinear_upsampled_normal.png",
    "sdf_bilinear_upsampled_motion.png",
    "sdf_bilinear_upsampled_depth.png",
];

fn example_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/upsampling")
}

fn pixel(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let offset = ((y * width + x) * 4) as usize;
    pixels[offset..offset + 4].try_into().unwrap()
}

fn deviation_from_neutral(motion: [u8; 4]) -> i32 {
    let red = i32::from(motion[0]) - 128;
    let green = i32::from(motion[1]) - 128;
    red * red + green * green
}

#[test]
fn upsampling_example_renders_color_normal_motion_and_depth() {
    let example = example_directory();
    for output in OUTPUT_IMAGES {
        let _ = fs::remove_file(example.join(output));
    }

    let graph = ComputeGraph::from_toml_file(example.join("sdf_bilinear_upsample.toml"))
        .expect("the upsampling example graph should parse");
    let mut execution = graph
        .execute_with_validation_layers(true)
        .expect("the upsampling example graph should execute");

    let images = [
        ("output", execution.read_image_rgba8("output")),
        ("output_normal", execution.read_image_rgba8("output_normal")),
        ("output_motion", execution.read_image_rgba8("output_motion")),
        ("output_depth", execution.read_image_rgba8("output_depth")),
    ];
    for (name, image) in &images {
        let (width, height, _) = image
            .as_ref()
            .unwrap_or_else(|error| panic!("example output `{name}` should be readable: {error}"));
        assert_eq!(
            (*width, *height),
            (WIDTH, HEIGHT),
            "example output `{name}` should match the declared output extent"
        );
    }

    let (_, _, color) = images[0].1.as_ref().unwrap();
    let (_, _, normal) = images[1].1.as_ref().unwrap();
    let (_, _, motion) = images[2].1.as_ref().unwrap();
    let (_, _, depth) = images[3].1.as_ref().unwrap();

    let plane = (2, 2);
    let sphere = (24, 15);

    assert_ne!(
        pixel(color, WIDTH, plane.0, plane.1),
        pixel(color, WIDTH, sphere.0, sphere.1),
        "the static plane and an animated sphere should be shaded differently"
    );

    assert_eq!(
        pixel(normal, WIDTH, plane.0, plane.1),
        [128, 128, 255, 255],
        "the unmoving plane should face the camera with a neutral normal"
    );
    assert_ne!(
        pixel(normal, WIDTH, plane.0, plane.1),
        pixel(normal, WIDTH, sphere.0, sphere.1),
        "the sphere normal should differ from the plane normal"
    );

    let plane_depth = pixel(depth, WIDTH, plane.0, plane.1)[0];
    let sphere_depth = pixel(depth, WIDTH, sphere.0, sphere.1)[0];
    assert!(
        plane_depth > sphere_depth,
        "the background plane ({plane_depth}) should be farther away than the sphere ({sphere_depth})"
    );

    let plane_motion = pixel(motion, WIDTH, plane.0, plane.1);
    let sphere_motion = pixel(motion, WIDTH, sphere.0, sphere.1);
    assert_ne!(
        [plane_motion[0], plane_motion[1]],
        [128, 128],
        "camera translation should produce a non-zero motion vector on the static plane"
    );
    assert!(
        deviation_from_neutral(sphere_motion) > deviation_from_neutral(plane_motion),
        "the animated sphere motion should exceed the camera-only plane motion"
    );
    for (x, y) in [plane, sphere] {
        let sample = pixel(motion, WIDTH, x, y);
        assert_eq!(
            (sample[2], sample[3]),
            (0, 255),
            "motion vectors should encode displacement in the red and green channels"
        );
    }

    for output in OUTPUT_IMAGES {
        let path = example.join(output);
        let image = image::open(&path)
            .unwrap_or_else(|error| panic!("example should write `{output}`: {error}"));
        assert_eq!(image.width(), WIDTH);
        assert_eq!(image.height(), HEIGHT);
        fs::remove_file(&path).expect("example output should be removable");
    }
}
