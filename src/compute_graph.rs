use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::CString,
    fmt, fs,
    path::{Path, PathBuf},
};

use ash::{Entry, vk};
use hassle_rs::compile_hlsl;
use serde::Deserialize;

use crate::{
    color_subresource_range, create_vulkan_instance, find_memory_type, select_compute_queue,
};

const MAX_RESOURCES: usize = 64;
const RESOURCE_TABLE_CAPACITY: usize = 16;

/// How a dispatch uses a graph resource.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AccessType {
    #[serde(alias = "r", alias = "read_only")]
    Read,
    #[serde(alias = "w", alias = "write_only")]
    Write,
    #[serde(alias = "readwrite", alias = "rw")]
    ReadWrite,
}

impl AccessType {
    fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

/// The resource kinds currently supported by the graph executor.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    Image,
    Buffer,
}

/// Vulkan image formats supported by graph image inputs and outputs.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
    #[serde(alias = "r8")]
    R8Unorm,
    #[serde(alias = "rg8", alias = "r8g8_unorm")]
    R8G8Unorm,
    #[serde(alias = "rgba8", alias = "rgba8_unorm", alias = "r8g8b8a8_unorm")]
    R8G8B8A8Unorm,
    #[serde(alias = "bgra8", alias = "bgra8_unorm", alias = "b8g8r8a8_unorm")]
    B8G8R8A8Unorm,
    #[serde(alias = "r32_float")]
    R32Sfloat,
    #[serde(alias = "rg32_float", alias = "r32g32_sfloat")]
    R32G32Sfloat,
    #[serde(alias = "rgba32_float", alias = "r32g32b32a32_sfloat")]
    R32G32B32A32Sfloat,
}

impl ImageFormat {
    fn vk_format(self) -> vk::Format {
        match self {
            Self::R8Unorm => vk::Format::R8_UNORM,
            Self::R8G8Unorm => vk::Format::R8G8_UNORM,
            Self::R8G8B8A8Unorm => vk::Format::R8G8B8A8_UNORM,
            Self::B8G8R8A8Unorm => vk::Format::B8G8R8A8_UNORM,
            Self::R32Sfloat => vk::Format::R32_SFLOAT,
            Self::R32G32Sfloat => vk::Format::R32G32_SFLOAT,
            Self::R32G32B32A32Sfloat => vk::Format::R32G32B32A32_SFLOAT,
        }
    }

    fn pixel_size(self) -> u64 {
        match self {
            Self::R8Unorm => 1,
            Self::R8G8Unorm => 2,
            Self::R8G8B8A8Unorm | Self::B8G8R8A8Unorm => 4,
            Self::R32Sfloat => 4,
            Self::R32G32Sfloat => 8,
            Self::R32G32B32A32Sfloat => 16,
        }
    }

    fn input_pixels(self, image: &image::DynamicImage) -> Vec<u8> {
        let rgba = image.to_rgba8();
        let mut pixels = Vec::with_capacity(
            rgba.width() as usize * rgba.height() as usize * self.pixel_size() as usize,
        );
        for pixel in rgba.pixels() {
            let [red, green, blue, alpha] = pixel.0;
            match self {
                Self::R8Unorm => pixels.push(red),
                Self::R8G8Unorm => pixels.extend_from_slice(&[red, green]),
                Self::R8G8B8A8Unorm => pixels.extend_from_slice(&[red, green, blue, alpha]),
                Self::B8G8R8A8Unorm => pixels.extend_from_slice(&[blue, green, red, alpha]),
                Self::R32Sfloat => {
                    pixels.extend_from_slice(&(f32::from(red) / 255.0).to_le_bytes())
                }
                Self::R32G32Sfloat => {
                    pixels.extend_from_slice(&(f32::from(red) / 255.0).to_le_bytes());
                    pixels.extend_from_slice(&(f32::from(green) / 255.0).to_le_bytes());
                }
                Self::R32G32B32A32Sfloat => {
                    for channel in [red, green, blue, alpha] {
                        pixels.extend_from_slice(&(f32::from(channel) / 255.0).to_le_bytes());
                    }
                }
            }
        }
        pixels
    }

    fn output_rgba8(self, data: &[u8]) -> Result<Vec<u8>, ComputeGraphError> {
        let channels = match self {
            Self::R8Unorm => 1,
            Self::R8G8Unorm => 2,
            Self::R8G8B8A8Unorm | Self::B8G8R8A8Unorm => 4,
            Self::R32Sfloat => 1,
            Self::R32G32Sfloat => 2,
            Self::R32G32B32A32Sfloat => 4,
        };
        let pixel_size = self.pixel_size() as usize;
        if !data.len().is_multiple_of(pixel_size) {
            return Err(ComputeGraphError::Invalid(format!(
                "image data length {} is not aligned to the `{self:?}` pixel size",
                data.len()
            )));
        }
        let mut pixels = Vec::with_capacity(data.len() / pixel_size * 4);
        for raw in data.chunks_exact(pixel_size) {
            let mut rgba = [0, 0, 0, 255];
            if channels <= 2 {
                rgba[0] = image_channel(raw, self, 0);
                rgba[1] = if channels == 2 {
                    image_channel(raw, self, 1)
                } else {
                    0
                };
            } else {
                rgba[0] = image_channel(raw, self, 0);
                rgba[1] = image_channel(raw, self, 1);
                rgba[2] = image_channel(raw, self, 2);
                rgba[3] = image_channel(raw, self, 3);
            }
            pixels.extend_from_slice(&rgba);
        }
        Ok(pixels)
    }
}

fn image_channel(data: &[u8], format: ImageFormat, channel: usize) -> u8 {
    match format {
        ImageFormat::R8Unorm | ImageFormat::R8G8Unorm => data[channel],
        ImageFormat::R8G8B8A8Unorm => data[channel],
        ImageFormat::B8G8R8A8Unorm => data[[2, 1, 0, 3][channel]],
        ImageFormat::R32Sfloat | ImageFormat::R32G32Sfloat | ImageFormat::R32G32B32A32Sfloat => {
            let start = channel * 4;
            (f32::from_le_bytes(data[start..start + 4].try_into().unwrap()) * 255.0)
                .clamp(0.0, 255.0) as u8
        }
    }
}

/// A resource declared in the `[resources.<name>]` TOML tables.
#[derive(Clone, Debug, Deserialize)]
pub struct ResourceDefinition {
    #[serde(rename = "type", alias = "kind")]
    pub kind: ResourceKind,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub extent: Option<[u32; 2]>,
    #[serde(default)]
    pub size: Option<u64>,
    /// Optional input path, resolved relative to the graph TOML file.
    #[serde(default)]
    pub input: Option<PathBuf>,
    /// GPU image format. Defaults to `r8g8b8a8_unorm` when no input is used.
    #[serde(default)]
    pub format: Option<ImageFormat>,
    /// Optional output path, resolved relative to the graph TOML file.
    #[serde(default)]
    pub output: Option<PathBuf>,
}

/// A resource use declared in a node's `bindings` array.
#[derive(Clone, Debug, Deserialize)]
pub struct ResourceBindingDefinition {
    #[serde(alias = "name")]
    pub resource: String,
    #[serde(alias = "usage", alias = "mode")]
    pub access: AccessType,
}

/// A compute dispatch declared in `[[nodes]]`.
#[derive(Clone, Debug, Deserialize)]
pub struct ComputeNodeDefinition {
    pub name: String,
    #[serde(alias = "shader_path", alias = "source", alias = "hlsl")]
    pub shader: PathBuf,
    pub kernel: String,
    #[serde(alias = "dispatch_size")]
    pub dispatch: [u32; 3],
    #[serde(default)]
    pub bindings: Vec<ResourceBindingDefinition>,
}

/// The directly deserializable form of a compute graph.
#[derive(Clone, Debug, Deserialize)]
pub struct ComputeGraphDefinition {
    #[serde(default)]
    pub resources: BTreeMap<String, ResourceDefinition>,
    #[serde(default)]
    pub nodes: Vec<ComputeNodeDefinition>,
}

/// Errors produced while loading or executing a compute graph.
#[derive(Debug)]
pub enum ComputeGraphError {
    Io(std::io::Error),
    Image(image::ImageError),
    Parse(toml::de::Error),
    Invalid(String),
    Vulkan(vk::Result),
    Shader(String),
}

impl fmt::Display for ComputeGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Image(error) => write!(formatter, "image error: {error}"),
            Self::Parse(error) => write!(formatter, "TOML error: {error}"),
            Self::Invalid(error) => formatter.write_str(error),
            Self::Vulkan(error) => write!(formatter, "Vulkan error: {error:?}"),
            Self::Shader(error) => write!(formatter, "shader compilation error: {error}"),
        }
    }
}

impl std::error::Error for ComputeGraphError {}

impl From<std::io::Error> for ComputeGraphError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<image::ImageError> for ComputeGraphError {
    fn from(error: image::ImageError) -> Self {
        Self::Image(error)
    }
}

impl From<toml::de::Error> for ComputeGraphError {
    fn from(error: toml::de::Error) -> Self {
        Self::Parse(error)
    }
}

impl From<vk::Result> for ComputeGraphError {
    fn from(error: vk::Result) -> Self {
        Self::Vulkan(error)
    }
}

#[derive(Clone, Debug)]
struct DependencyGraph {
    prerequisites: Vec<Vec<usize>>,
}

/// A validated, data-driven compute graph.
#[derive(Clone, Debug)]
pub struct ComputeGraph {
    definition: ComputeGraphDefinition,
    base_dir: PathBuf,
    slots: BTreeMap<String, u32>,
    dependencies: DependencyGraph,
}

impl ComputeGraph {
    /// Parses and validates a graph. Shader files are resolved relative to the
    /// current working directory when this constructor is used.
    pub fn from_toml(source: &str) -> Result<Self, ComputeGraphError> {
        Self::from_toml_with_base(source, PathBuf::from("."))
    }

    /// Loads and validates a graph file. Relative shader paths are resolved
    /// relative to the TOML file.
    pub fn from_toml_file(path: impl AsRef<Path>) -> Result<Self, ComputeGraphError> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)?;
        let base_dir = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        Self::from_toml_with_base(&source, base_dir)
    }

    fn from_toml_with_base(source: &str, base_dir: PathBuf) -> Result<Self, ComputeGraphError> {
        let definition = toml::from_str::<ComputeGraphDefinition>(source)?;
        Self::from_definition_with_base(definition, base_dir)
    }

    fn from_definition_with_base(
        definition: ComputeGraphDefinition,
        base_dir: PathBuf,
    ) -> Result<Self, ComputeGraphError> {
        let mut slots = BTreeMap::new();
        for (index, name) in definition.resources.keys().enumerate() {
            if index >= MAX_RESOURCES {
                return Err(ComputeGraphError::Invalid(format!(
                    "compute graph declares more than {MAX_RESOURCES} resources"
                )));
            }
            slots.insert(name.clone(), index as u32);
        }

        for (name, resource) in &definition.resources {
            match resource.kind {
                ResourceKind::Image => {
                    if resource.input.is_some()
                        && (resource.width.is_some()
                            || resource.height.is_some()
                            || resource.extent.is_some())
                    {
                        return Err(ComputeGraphError::Invalid(format!(
                            "image resource `{name}` must not specify dimensions when input is used"
                        )));
                    }
                    let input_dimensions = if let Some(input) = &resource.input {
                        if resource.format.is_none() {
                            return Err(ComputeGraphError::Invalid(format!(
                                "image resource `{name}` must specify a format when input is used"
                            )));
                        }
                        let input_path = base_dir.join(input);
                        let input_image = image::open(&input_path)?;
                        if input_image.width() == 0 || input_image.height() == 0 {
                            return Err(ComputeGraphError::Invalid(format!(
                                "image resource `{name}` input `{}` has zero dimensions",
                                input_path.display()
                            )));
                        }
                        Some((input_image.width(), input_image.height()))
                    } else {
                        None
                    };
                    let width = resource
                        .width
                        .or(resource.extent.map(|extent| extent[0]))
                        .or(input_dimensions.map(|dimensions| dimensions.0))
                        .unwrap_or(0);
                    let height = resource
                        .height
                        .or(resource.extent.map(|extent| extent[1]))
                        .or(input_dimensions.map(|dimensions| dimensions.1))
                        .unwrap_or(0);
                    if width == 0 || height == 0 {
                        return Err(ComputeGraphError::Invalid(format!(
                            "image resource `{name}` must have non-zero width and height"
                        )));
                    }
                    if let Some(output) = &resource.output
                        && !output
                            .extension()
                            .map(|extension| {
                                extension.to_string_lossy().eq_ignore_ascii_case("png")
                            })
                            .unwrap_or(false)
                    {
                        return Err(ComputeGraphError::Invalid(format!(
                            "image resource `{name}` output path `{}` must use a .png extension",
                            output.display()
                        )));
                    }
                }
                ResourceKind::Buffer => {
                    if resource.input.is_some() && resource.size.is_some() {
                        return Err(ComputeGraphError::Invalid(format!(
                            "buffer resource `{name}` must not specify size when input is used"
                        )));
                    }
                    let size = match &resource.input {
                        Some(input) => fs::metadata(base_dir.join(input))?.len(),
                        None => resource.size.unwrap_or(0),
                    };
                    if size == 0 {
                        return Err(ComputeGraphError::Invalid(format!(
                            "buffer resource `{name}` must have a non-zero size"
                        )));
                    }
                }
            }
        }

        let mut prerequisites = vec![Vec::new(); definition.nodes.len()];
        let mut last_writer = BTreeMap::<String, usize>::new();
        let mut readers = BTreeMap::<String, BTreeSet<usize>>::new();
        for (node_index, node) in definition.nodes.iter().enumerate() {
            if node.name.is_empty() {
                return Err(ComputeGraphError::Invalid(format!(
                    "node {node_index} has an empty name"
                )));
            }
            if node.kernel.is_empty() {
                return Err(ComputeGraphError::Invalid(format!(
                    "node `{}` has an empty kernel name",
                    node.name
                )));
            }
            if node.dispatch.contains(&0) {
                return Err(ComputeGraphError::Invalid(format!(
                    "node `{}` has a zero dispatch dimension",
                    node.name
                )));
            }
            let mut bound = BTreeSet::new();
            for binding in &node.bindings {
                let resource = definition.resources.get(&binding.resource).ok_or_else(|| {
                    ComputeGraphError::Invalid(format!(
                        "node `{}` refers to undeclared resource `{}`",
                        node.name, binding.resource
                    ))
                })?;
                if !bound.insert(binding.resource.clone()) {
                    return Err(ComputeGraphError::Invalid(format!(
                        "node `{}` binds resource `{}` more than once",
                        node.name, binding.resource
                    )));
                }
                if binding.access.reads() {
                    if let Some(writer) = last_writer.get(&binding.resource) {
                        prerequisites[node_index].push(*writer);
                    }
                    readers
                        .entry(binding.resource.clone())
                        .or_default()
                        .insert(node_index);
                }
                if binding.access.writes() {
                    if let Some(writer) = last_writer.get(&binding.resource) {
                        prerequisites[node_index].push(*writer);
                    }
                    if let Some(previous_readers) = readers.get(&binding.resource) {
                        prerequisites[node_index].extend(
                            previous_readers
                                .iter()
                                .copied()
                                .filter(|reader| *reader != node_index),
                        );
                    }
                    readers.remove(&binding.resource);
                    last_writer.insert(binding.resource.clone(), node_index);
                }
                let _ = resource;
            }
            if node.bindings.len() > RESOURCE_TABLE_CAPACITY {
                return Err(ComputeGraphError::Invalid(format!(
                    "node `{}` binds more than {RESOURCE_TABLE_CAPACITY} resources",
                    node.name
                )));
            }
            prerequisites[node_index].sort_unstable();
            prerequisites[node_index].dedup();
        }

        Ok(Self {
            definition,
            base_dir,
            slots,
            dependencies: DependencyGraph { prerequisites },
        })
    }

    pub fn definition(&self) -> &ComputeGraphDefinition {
        &self.definition
    }

    pub fn resource_slot(&self, name: &str) -> Option<u32> {
        self.slots.get(name).copied()
    }

    pub fn dependencies(&self, node: usize) -> Option<&[usize]> {
        self.dependencies.prerequisites.get(node).map(Vec::as_slice)
    }

    /// Creates all declared resources, compiles the node shaders, and executes
    /// the graph in dependency order on a compute queue.
    pub fn execute(&self) -> Result<ComputeGraphExecution, ComputeGraphError> {
        self.execute_with_validation_layers(false)
    }

    pub fn execute_with_validation_layers(
        &self,
        enable_validation_layers: bool,
    ) -> Result<ComputeGraphExecution, ComputeGraphError> {
        let mut runtime = GraphRuntime::new(self, enable_validation_layers)?;
        runtime.execute(self)?;
        let mut execution = ComputeGraphExecution { runtime };
        execution.write_outputs(self)?;
        Ok(execution)
    }
}

/// Resources and GPU state retained after graph execution for inspection.
pub struct ComputeGraphExecution {
    runtime: GraphRuntime,
}

impl ComputeGraphExecution {
    pub fn resource_slot(&self, name: &str) -> Option<u32> {
        self.runtime
            .resources
            .get(name)
            .map(|resource| resource.slot)
    }

    /// Copies an image resource to host memory as tightly packed RGBA8 pixels.
    pub fn read_image_rgba8(
        &mut self,
        name: &str,
    ) -> Result<(u32, u32, Vec<u8>), ComputeGraphError> {
        self.runtime.read_image(name)
    }

    /// Copies a buffer resource to host memory.
    pub fn read_buffer(&mut self, name: &str) -> Result<Vec<u8>, ComputeGraphError> {
        self.runtime.read_buffer(name)
    }

    fn write_outputs(&mut self, graph: &ComputeGraph) -> Result<(), ComputeGraphError> {
        for (name, resource) in &graph.definition.resources {
            let Some(output) = &resource.output else {
                continue;
            };
            let output = graph.base_dir.join(output);
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            match resource.kind {
                ResourceKind::Image => {
                    let (width, height, pixels) = self.runtime.read_image(name)?;
                    image::save_buffer_with_format(
                        &output,
                        &pixels,
                        width,
                        height,
                        image::ColorType::Rgba8,
                        image::ImageFormat::Png,
                    )?;
                }
                ResourceKind::Buffer => {
                    let bytes = self.runtime.read_buffer(name)?;
                    fs::write(output, bytes)?;
                }
            }
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ResourceTablePushConstants {
    slots: [u32; RESOURCE_TABLE_CAPACITY],
}

struct GraphImage {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    width: u32,
    height: u32,
    format: ImageFormat,
    layout: vk::ImageLayout,
    access: vk::AccessFlags,
}

struct GraphBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    access: vk::AccessFlags,
}

enum GraphResource {
    Image(GraphImage),
    Buffer(GraphBuffer),
}

struct RuntimeResource {
    slot: u32,
    resource: GraphResource,
}

struct GraphNode {
    module: vk::ShaderModule,
    pipeline: vk::Pipeline,
}

struct GraphBindlessTable {
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

struct ImmutableSamplerTable {
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    sampler: vk::Sampler,
}

impl ImmutableSamplerTable {
    fn new(device: &ash::Device) -> Result<Self, ComputeGraphError> {
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(0.0);
        let sampler = unsafe { device.create_sampler(&sampler_info, None)? };
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .immutable_samplers(std::slice::from_ref(&sampler));
        let layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(std::slice::from_ref(&binding));
        let layout = match unsafe { device.create_descriptor_set_layout(&layout_info, None) } {
            Ok(layout) => layout,
            Err(error) => {
                unsafe { device.destroy_sampler(sampler, None) };
                return Err(error.into());
            }
        };
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLER)
            .descriptor_count(1);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(std::slice::from_ref(&pool_size));
        let pool = match unsafe { device.create_descriptor_pool(&pool_info, None) } {
            Ok(pool) => pool,
            Err(error) => {
                unsafe {
                    device.destroy_descriptor_set_layout(layout, None);
                    device.destroy_sampler(sampler, None);
                }
                return Err(error.into());
            }
        };
        let set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(std::slice::from_ref(&layout));
        let set = match unsafe { device.allocate_descriptor_sets(&set_info) } {
            Ok(sets) => sets[0],
            Err(error) => {
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                    device.destroy_descriptor_set_layout(layout, None);
                    device.destroy_sampler(sampler, None);
                }
                return Err(error.into());
            }
        };
        Ok(Self {
            layout,
            pool,
            set,
            sampler,
        })
    }
}

struct GraphRuntime {
    instance: crate::VulkanInstance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    bindless: GraphBindlessTable,
    sampler: ImmutableSamplerTable,
    resources: BTreeMap<String, RuntimeResource>,
    nodes: Vec<GraphNode>,
    pipeline_layout: vk::PipelineLayout,
    command_pool: vk::CommandPool,
}

impl GraphRuntime {
    fn new(
        graph: &ComputeGraph,
        enable_validation_layers: bool,
    ) -> Result<Self, ComputeGraphError> {
        let entry = unsafe { Entry::load() }
            .map_err(|error| ComputeGraphError::Invalid(error.to_string()))?;
        let app_name = CString::new("ai-vk-compute-graph").expect("static application name");
        let instance = create_vulkan_instance(
            entry,
            &app_name,
            vk::API_VERSION_1_2,
            enable_validation_layers,
        )
        .map_err(|error| ComputeGraphError::Invalid(error.to_string()))?;
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let (physical_device, queue_family_index) =
            select_compute_queue(&instance, &physical_devices)
                .map_err(|error| ComputeGraphError::Invalid(error.to_string()))?;

        let mut supported_descriptor = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
        let mut supported =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut supported_descriptor);
        unsafe { instance.get_physical_device_features2(physical_device, &mut supported) };
        if supported
            .features
            .shader_storage_image_array_dynamic_indexing
            == vk::FALSE
            || supported
                .features
                .shader_sampled_image_array_dynamic_indexing
                == vk::FALSE
            || supported
                .features
                .shader_storage_buffer_array_dynamic_indexing
                == vk::FALSE
            || supported_descriptor.descriptor_binding_partially_bound == vk::FALSE
        {
            return Err(ComputeGraphError::Invalid(
                "selected Vulkan device lacks bindless compute graph features".into(),
            ));
        }
        let core_features = vk::PhysicalDeviceFeatures {
            shader_storage_image_array_dynamic_indexing: vk::TRUE,
            shader_sampled_image_array_dynamic_indexing: vk::TRUE,
            shader_storage_buffer_array_dynamic_indexing: vk::TRUE,
            ..Default::default()
        };
        let mut descriptor_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
            .descriptor_binding_partially_bound(true);
        let queue_priority = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priority);
        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_features(&core_features);
        device_info = device_info.push_next(&mut descriptor_features);
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let bindless = create_graph_bindless_table(&device)?;
        let sampler = ImmutableSamplerTable::new(&device)
            .map_err(|error| ComputeGraphError::Invalid(error.to_string()))?;

        let mut resources = BTreeMap::new();
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        for (name, definition) in &graph.definition.resources {
            let slot = graph.slots[name];
            let resource = match definition.kind {
                ResourceKind::Image => {
                    let (width, height) = image_dimensions(graph, name, definition)?;
                    let format = definition.format.unwrap_or(ImageFormat::R8G8B8A8Unorm);
                    let image_info = vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format.vk_format())
                        .extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(
                            vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::SAMPLED
                                | vk::ImageUsageFlags::TRANSFER_SRC
                                | vk::ImageUsageFlags::TRANSFER_DST,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED);
                    let (image, memory) =
                        create_graph_image(&device, &image_info, &memory_properties)?;
                    let view_info = vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format.vk_format())
                        .subresource_range(color_subresource_range());
                    let view = unsafe { device.create_image_view(&view_info, None)? };
                    add_graph_image(&device, &bindless, slot, view)?;
                    GraphResource::Image(GraphImage {
                        image,
                        view,
                        memory,
                        width,
                        height,
                        format,
                        layout: vk::ImageLayout::UNDEFINED,
                        access: vk::AccessFlags::empty(),
                    })
                }
                ResourceKind::Buffer => {
                    let size = buffer_size(graph, definition)?;
                    let buffer_info = vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(
                            vk::BufferUsageFlags::STORAGE_BUFFER
                                | vk::BufferUsageFlags::TRANSFER_SRC
                                | vk::BufferUsageFlags::TRANSFER_DST,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE);
                    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
                    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
                    let (memory_type, _) = find_memory_type(
                        &memory_properties,
                        requirements.memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        vk::MemoryPropertyFlags::empty(),
                    )
                    .or_else(|_| {
                        find_memory_type(
                            &memory_properties,
                            requirements.memory_type_bits,
                            vk::MemoryPropertyFlags::empty(),
                            vk::MemoryPropertyFlags::empty(),
                        )
                    })
                    .map_err(|error| ComputeGraphError::Invalid(error.to_string()))?;
                    let allocate = vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(memory_type);
                    let memory = unsafe { device.allocate_memory(&allocate, None)? };
                    unsafe { device.bind_buffer_memory(buffer, memory, 0)? };
                    add_graph_buffer(&device, &bindless, slot, buffer)?;
                    GraphResource::Buffer(GraphBuffer {
                        buffer,
                        memory,
                        size,
                        access: vk::AccessFlags::empty(),
                    })
                }
            };
            resources.insert(name.clone(), RuntimeResource { slot, resource });
        }

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<ResourceTablePushConstants>() as u32);
        let layouts = [sampler.layout, bindless.layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(std::slice::from_ref(&push_range));
        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        let include_path = graph.base_dir.to_string_lossy().into_owned();
        let include_argument = format!("-I{include_path}");
        let mut nodes = Vec::with_capacity(graph.definition.nodes.len());
        for node in &graph.definition.nodes {
            let shader_path = graph.base_dir.join(&node.shader);
            let source = fs::read_to_string(&shader_path)?;
            let spirv = compile_hlsl(
                &shader_path.to_string_lossy(),
                &source,
                &node.kernel,
                "cs_6_0",
                &["-spirv", include_argument.as_str()],
                &[],
            )
            .map_err(|error| ComputeGraphError::Shader(error.to_string()))?;
            if !spirv.len().is_multiple_of(4) {
                return Err(ComputeGraphError::Shader(format!(
                    "shader `{}` produced invalid SPIR-V",
                    shader_path.display()
                )));
            }
            let code: Vec<u32> = spirv
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect();
            let module = unsafe {
                device.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&code),
                    None,
                )?
            };
            let kernel = CString::new(node.kernel.as_str()).map_err(|_| {
                ComputeGraphError::Invalid(format!(
                    "node `{}` has an invalid kernel name",
                    node.name
                ))
            })?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(&kernel);
            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout);
            let pipeline = unsafe {
                device
                    .create_compute_pipelines(
                        vk::PipelineCache::null(),
                        std::slice::from_ref(&pipeline_info),
                        None,
                    )
                    .map_err(|(_, error)| error)?[0]
            };
            nodes.push(GraphNode { module, pipeline });
        }

        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(queue_family_index);
        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None)? };
        let mut runtime = Self {
            instance,
            device,
            physical_device,
            queue,
            bindless,
            sampler,
            resources,
            nodes,
            pipeline_layout,
            command_pool,
        };
        runtime.upload_inputs(graph)?;
        Ok(runtime)
    }

    fn upload_inputs(&mut self, graph: &ComputeGraph) -> Result<(), ComputeGraphError> {
        if !graph
            .definition
            .resources
            .values()
            .any(|resource| resource.input.is_some())
        {
            return Ok(());
        }

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command = unsafe { self.device.allocate_command_buffers(&allocate_info)?[0] };
        unsafe {
            self.device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())?;
        }
        let properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        let mut staging = Vec::new();
        for (name, definition) in &graph.definition.resources {
            let Some(input) = &definition.input else {
                continue;
            };
            let input_path = graph.base_dir.join(input);
            let bytes = match definition.kind {
                ResourceKind::Image => {
                    let image = image::open(&input_path)?;
                    let format = definition.format.ok_or_else(|| {
                        ComputeGraphError::Invalid(format!(
                            "image resource `{name}` must specify a format when input is used"
                        ))
                    })?;
                    format.input_pixels(&image)
                }
                ResourceKind::Buffer => fs::read(&input_path)?,
            };
            let size = u64::try_from(bytes.len()).map_err(|_| {
                ComputeGraphError::Invalid(format!("input `{}` is too large", input_path.display()))
            })?;
            let buffer_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = unsafe { self.device.create_buffer(&buffer_info, None)? };
            let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
            let (memory_type, flags) = find_memory_type(
                &properties,
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .map_err(|error| ComputeGraphError::Invalid(error.to_string()))?;
            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type);
            let memory = unsafe { self.device.allocate_memory(&allocate, None)? };
            unsafe { self.device.bind_buffer_memory(buffer, memory, 0)? };
            let mapped = unsafe {
                self.device
                    .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?
            };
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
            }
            if !flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
                let range = vk::MappedMemoryRange::default()
                    .memory(memory)
                    .offset(0)
                    .size(size);
                unsafe {
                    self.device
                        .flush_mapped_memory_ranges(std::slice::from_ref(&range))?;
                }
            }
            unsafe { self.device.unmap_memory(memory) };

            let runtime_resource = self.resources.get_mut(name).ok_or_else(|| {
                ComputeGraphError::Invalid(format!("missing runtime resource `{name}`"))
            })?;
            match &mut runtime_resource.resource {
                GraphResource::Image(image) => {
                    let barrier = vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .image(image.image)
                        .subresource_range(color_subresource_range());
                    let region = vk::BufferImageCopy::default()
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(0)
                                .base_array_layer(0)
                                .layer_count(1),
                        )
                        .image_extent(vk::Extent3D {
                            width: image.width,
                            height: image.height,
                            depth: 1,
                        });
                    unsafe {
                        self.device.cmd_pipeline_barrier(
                            command,
                            vk::PipelineStageFlags::TOP_OF_PIPE,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            std::slice::from_ref(&barrier),
                        );
                        self.device.cmd_copy_buffer_to_image(
                            command,
                            buffer,
                            image.image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            std::slice::from_ref(&region),
                        );
                    }
                    image.layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                    image.access = vk::AccessFlags::TRANSFER_WRITE;
                }
                GraphResource::Buffer(graph_buffer) => {
                    let barrier = vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .buffer(graph_buffer.buffer)
                        .offset(0)
                        .size(graph_buffer.size);
                    unsafe {
                        self.device.cmd_pipeline_barrier(
                            command,
                            vk::PipelineStageFlags::TOP_OF_PIPE,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &[],
                            std::slice::from_ref(&barrier),
                            &[],
                        );
                        self.device.cmd_copy_buffer(
                            command,
                            buffer,
                            graph_buffer.buffer,
                            std::slice::from_ref(&vk::BufferCopy::default().size(size)),
                        );
                    }
                    graph_buffer.access = vk::AccessFlags::TRANSFER_WRITE;
                }
            }
            staging.push((buffer, memory));
        }
        unsafe {
            self.device.end_command_buffer(command)?;
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command));
            self.device.queue_submit(
                self.queue,
                std::slice::from_ref(&submit),
                vk::Fence::null(),
            )?;
            self.device.queue_wait_idle(self.queue)?;
            for (buffer, memory) in staging {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
        }
        Ok(())
    }

    fn execute(&mut self, graph: &ComputeGraph) -> Result<(), ComputeGraphError> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { self.device.allocate_command_buffers(&allocate_info)?[0] };
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?
        };
        let descriptor_sets = [self.sampler.set, self.bindless.set];
        let mut completed = vec![false; graph.definition.nodes.len()];
        let mut completed_count = 0;
        while completed_count < completed.len() {
            let node_index = (0..completed.len())
                .find(|index| {
                    !completed[*index]
                        && graph.dependencies.prerequisites[*index]
                            .iter()
                            .all(|dependency| completed[*dependency])
                })
                .ok_or_else(|| {
                    ComputeGraphError::Invalid("compute graph contains a dependency cycle".into())
                })?;
            let node = &graph.definition.nodes[node_index];
            self.record_barriers(command_buffer, node)?;
            unsafe {
                self.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.nodes[node_index].pipeline,
                );
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &descriptor_sets,
                    &[],
                );
                let mut table = ResourceTablePushConstants {
                    slots: [0; RESOURCE_TABLE_CAPACITY],
                };
                if node.bindings.len() > RESOURCE_TABLE_CAPACITY {
                    return Err(ComputeGraphError::Invalid(format!(
                        "node `{}` binds more than {RESOURCE_TABLE_CAPACITY} resources",
                        node.name
                    )));
                }
                for (index, binding) in node.bindings.iter().enumerate() {
                    table.slots[index] = self.resources[&binding.resource].slot;
                }
                let bytes = std::slice::from_raw_parts(
                    (&table as *const ResourceTablePushConstants).cast::<u8>(),
                    std::mem::size_of::<ResourceTablePushConstants>(),
                );
                self.device.cmd_push_constants(
                    command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytes,
                );
                self.device.cmd_dispatch(
                    command_buffer,
                    node.dispatch[0],
                    node.dispatch[1],
                    node.dispatch[2],
                );
            }
            completed[node_index] = true;
            completed_count += 1;
        }
        unsafe {
            self.device.end_command_buffer(command_buffer)?;
            let submit =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
            self.device.queue_submit(
                self.queue,
                std::slice::from_ref(&submit),
                vk::Fence::null(),
            )?;
            self.device.queue_wait_idle(self.queue)?;
        }
        Ok(())
    }

    fn record_barriers(
        &mut self,
        command_buffer: vk::CommandBuffer,
        node: &ComputeNodeDefinition,
    ) -> Result<(), ComputeGraphError> {
        let mut image_barriers = Vec::new();
        let mut buffer_barriers = Vec::new();
        for binding in &node.bindings {
            let runtime_resource = self.resources.get_mut(&binding.resource).ok_or_else(|| {
                ComputeGraphError::Invalid(format!(
                    "missing runtime resource `{}`",
                    binding.resource
                ))
            })?;
            match &mut runtime_resource.resource {
                GraphResource::Image(image) => {
                    let new_layout = if binding.access.writes() {
                        vk::ImageLayout::GENERAL
                    } else {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    };
                    let needs_barrier = image.layout != new_layout
                        || !image.access.is_empty()
                        || binding.access.writes();
                    if needs_barrier {
                        image_barriers.push(
                            vk::ImageMemoryBarrier::default()
                                .src_access_mask(image.access)
                                .dst_access_mask(access_flags(binding.access))
                                .old_layout(image.layout)
                                .new_layout(new_layout)
                                .image(image.image)
                                .subresource_range(color_subresource_range()),
                        );
                    }
                    image.layout = new_layout;
                    image.access = access_flags(binding.access);
                }
                GraphResource::Buffer(buffer) => {
                    let needs_barrier = !buffer.access.is_empty() || binding.access.writes();
                    if needs_barrier {
                        buffer_barriers.push(
                            vk::BufferMemoryBarrier::default()
                                .src_access_mask(buffer.access)
                                .dst_access_mask(access_flags(binding.access))
                                .buffer(buffer.buffer)
                                .offset(0)
                                .size(buffer.size),
                        );
                    }
                    buffer.access = access_flags(binding.access);
                }
            }
        }
        if !image_barriers.is_empty() || !buffer_barriers.is_empty() {
            let mut source_stage = vk::PipelineStageFlags::empty();
            for access in image_barriers
                .iter()
                .map(|barrier| barrier.src_access_mask)
                .chain(
                    buffer_barriers
                        .iter()
                        .map(|barrier| barrier.src_access_mask),
                )
            {
                if access
                    .intersects(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::TRANSFER_READ)
                {
                    source_stage |= vk::PipelineStageFlags::TRANSFER;
                } else if !access.is_empty() {
                    source_stage |= vk::PipelineStageFlags::COMPUTE_SHADER;
                } else {
                    source_stage |= vk::PipelineStageFlags::TOP_OF_PIPE;
                }
            }
            unsafe {
                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    source_stage,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &buffer_barriers,
                    &image_barriers,
                );
            }
        }
        Ok(())
    }

    fn read_image(&mut self, name: &str) -> Result<(u32, u32, Vec<u8>), ComputeGraphError> {
        let runtime_resource = self.resources.get_mut(name).ok_or_else(|| {
            ComputeGraphError::Invalid(format!("unknown graph resource `{name}`"))
        })?;
        let GraphResource::Image(image) = &mut runtime_resource.resource else {
            return Err(ComputeGraphError::Invalid(format!(
                "resource `{name}` is not an image"
            )));
        };
        let raw_byte_count = u64::from(image.width)
            .checked_mul(u64::from(image.height))
            .and_then(|pixels| pixels.checked_mul(image.format.pixel_size()))
            .ok_or_else(|| {
                ComputeGraphError::Invalid("image staging buffer is too large".into())
            })?;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(raw_byte_count)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging = unsafe { self.device.create_buffer(&buffer_info, None)? };
        let requirements = unsafe { self.device.get_buffer_memory_requirements(staging) };
        let memory_properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        let (memory_type, flags) = find_memory_type(
            &memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .map_err(|error| ComputeGraphError::Invalid(error.to_string()))?;
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = unsafe { self.device.allocate_memory(&allocate, None)? };
        unsafe { self.device.bind_buffer_memory(staging, memory, 0)? };
        let command = unsafe {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            self.device.allocate_command_buffers(&info)?[0]
        };
        unsafe {
            self.device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())?;
            let barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(image.access)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(image.layout)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(image.image)
                .subresource_range(color_subresource_range());
            self.device.cmd_pipeline_barrier(
                command,
                if image
                    .access
                    .intersects(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)
                {
                    vk::PipelineStageFlags::TRANSFER
                } else if image.access.is_empty() {
                    vk::PipelineStageFlags::TOP_OF_PIPE
                } else {
                    vk::PipelineStageFlags::COMPUTE_SHADER
                },
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );
            let region = vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: image.width,
                    height: image.height,
                    depth: 1,
                });
            self.device.cmd_copy_image_to_buffer(
                command,
                image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging,
                std::slice::from_ref(&region),
            );
            self.device.end_command_buffer(command)?;
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command));
            self.device.queue_submit(
                self.queue,
                std::slice::from_ref(&submit),
                vk::Fence::null(),
            )?;
            self.device.queue_wait_idle(self.queue)?;
        }
        image.layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        image.access = vk::AccessFlags::TRANSFER_READ;
        let mapped = unsafe {
            self.device
                .map_memory(memory, 0, raw_byte_count, vk::MemoryMapFlags::empty())?
        };
        if !flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
            let range = vk::MappedMemoryRange::default()
                .memory(memory)
                .offset(0)
                .size(raw_byte_count);
            unsafe {
                self.device
                    .invalidate_mapped_memory_ranges(std::slice::from_ref(&range))?
            };
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(mapped.cast::<u8>(), raw_byte_count as usize).to_vec()
        };
        unsafe {
            self.device.unmap_memory(memory);
            self.device.destroy_buffer(staging, None);
            self.device.free_memory(memory, None);
        }
        let pixels = image.format.output_rgba8(&bytes)?;
        Ok((image.width, image.height, pixels))
    }

    fn read_buffer(&mut self, name: &str) -> Result<Vec<u8>, ComputeGraphError> {
        let runtime_resource = self.resources.get_mut(name).ok_or_else(|| {
            ComputeGraphError::Invalid(format!("unknown graph resource `{name}`"))
        })?;
        let GraphResource::Buffer(buffer) = &mut runtime_resource.resource else {
            return Err(ComputeGraphError::Invalid(format!(
                "resource `{name}` is not a buffer"
            )));
        };
        let staging_info = vk::BufferCreateInfo::default()
            .size(buffer.size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging = unsafe { self.device.create_buffer(&staging_info, None)? };
        let requirements = unsafe { self.device.get_buffer_memory_requirements(staging) };
        let properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        let (memory_type, flags) = find_memory_type(
            &properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .map_err(|error| ComputeGraphError::Invalid(error.to_string()))?;
        let memory = unsafe {
            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type);
            self.device.allocate_memory(&allocate, None)?
        };
        unsafe { self.device.bind_buffer_memory(staging, memory, 0)? };
        let command = unsafe {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            self.device.allocate_command_buffers(&info)?[0]
        };
        unsafe {
            self.device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())?;
            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(buffer.access)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .buffer(buffer.buffer)
                .offset(0)
                .size(buffer.size);
            let source_stage = if buffer
                .access
                .intersects(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)
            {
                vk::PipelineStageFlags::TRANSFER
            } else if buffer.access.is_empty() {
                vk::PipelineStageFlags::TOP_OF_PIPE
            } else {
                vk::PipelineStageFlags::COMPUTE_SHADER
            };
            self.device.cmd_pipeline_barrier(
                command,
                source_stage,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&barrier),
                &[],
            );
            let region = vk::BufferCopy::default().size(buffer.size);
            self.device.cmd_copy_buffer(
                command,
                buffer.buffer,
                staging,
                std::slice::from_ref(&region),
            );
            self.device.end_command_buffer(command)?;
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command));
            self.device.queue_submit(
                self.queue,
                std::slice::from_ref(&submit),
                vk::Fence::null(),
            )?;
            self.device.queue_wait_idle(self.queue)?;
        }
        buffer.access = vk::AccessFlags::TRANSFER_READ;
        let mapped = unsafe {
            self.device
                .map_memory(memory, 0, buffer.size, vk::MemoryMapFlags::empty())?
        };
        if !flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
            let range = vk::MappedMemoryRange::default()
                .memory(memory)
                .offset(0)
                .size(buffer.size);
            unsafe {
                self.device
                    .invalidate_mapped_memory_ranges(std::slice::from_ref(&range))?
            };
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(mapped.cast::<u8>(), buffer.size as usize).to_vec()
        };
        unsafe {
            self.device.unmap_memory(memory);
            self.device.destroy_buffer(staging, None);
            self.device.free_memory(memory, None);
        }
        Ok(bytes)
    }
}

impl Drop for GraphRuntime {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            for node in self.nodes.drain(..) {
                self.device.destroy_pipeline(node.pipeline, None);
                self.device.destroy_shader_module(node.module, None);
            }
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            for runtime_resource in self.resources.values() {
                match &runtime_resource.resource {
                    GraphResource::Image(image) => {
                        self.device.destroy_image_view(image.view, None);
                        self.device.destroy_image(image.image, None);
                        self.device.free_memory(image.memory, None);
                    }
                    GraphResource::Buffer(buffer) => {
                        self.device.destroy_buffer(buffer.buffer, None);
                        self.device.free_memory(buffer.memory, None);
                    }
                }
            }
            self.device
                .destroy_descriptor_pool(self.bindless.pool, None);
            self.device
                .destroy_descriptor_set_layout(self.bindless.layout, None);
            self.device.destroy_descriptor_pool(self.sampler.pool, None);
            self.device
                .destroy_descriptor_set_layout(self.sampler.layout, None);
            self.device.destroy_sampler(self.sampler.sampler, None);
            self.device.destroy_device(None);
        }
    }
}

fn access_flags(access: AccessType) -> vk::AccessFlags {
    match (access.reads(), access.writes()) {
        (true, false) => vk::AccessFlags::SHADER_READ,
        (false, true) => vk::AccessFlags::SHADER_WRITE,
        (true, true) => vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        (false, false) => vk::AccessFlags::empty(),
    }
}

fn image_dimensions(
    graph: &ComputeGraph,
    name: &str,
    definition: &ResourceDefinition,
) -> Result<(u32, u32), ComputeGraphError> {
    if let Some(input) = &definition.input {
        let image = image::open(graph.base_dir.join(input))?;
        return Ok((image.width(), image.height()));
    }
    let extent = definition.extent.unwrap_or([
        definition.width.unwrap_or(0),
        definition.height.unwrap_or(0),
    ]);
    let width = definition.width.unwrap_or(extent[0]);
    let height = definition.height.unwrap_or(extent[1]);
    if width == 0 || height == 0 {
        return Err(ComputeGraphError::Invalid(format!(
            "image resource `{name}` must have non-zero width and height"
        )));
    }
    Ok((width, height))
}

fn buffer_size(
    graph: &ComputeGraph,
    definition: &ResourceDefinition,
) -> Result<u64, ComputeGraphError> {
    match &definition.input {
        Some(input) => Ok(fs::metadata(graph.base_dir.join(input))?.len()),
        None => definition.size.ok_or_else(|| {
            ComputeGraphError::Invalid("buffer resource is missing its size".into())
        }),
    }
}

fn create_graph_bindless_table(
    device: &ash::Device,
) -> Result<GraphBindlessTable, ComputeGraphError> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(MAX_RESOURCES as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(MAX_RESOURCES as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(MAX_RESOURCES as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let flags = [
        vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        vk::DescriptorBindingFlags::PARTIALLY_BOUND,
    ];
    let mut binding_flags =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
    let mut layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    layout_info = layout_info.push_next(&mut binding_flags);
    let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(MAX_RESOURCES as u32),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(MAX_RESOURCES as u32),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(MAX_RESOURCES as u32),
    ];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let pool = match unsafe { device.create_descriptor_pool(&pool_info, None) } {
        Ok(pool) => pool,
        Err(error) => {
            unsafe { device.destroy_descriptor_set_layout(layout, None) };
            return Err(error.into());
        }
    };
    let set_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(std::slice::from_ref(&layout));
    let set = match unsafe { device.allocate_descriptor_sets(&set_info) } {
        Ok(sets) => sets[0],
        Err(error) => {
            unsafe {
                device.destroy_descriptor_pool(pool, None);
                device.destroy_descriptor_set_layout(layout, None);
            }
            return Err(error.into());
        }
    };
    Ok(GraphBindlessTable { layout, pool, set })
}

fn add_graph_image(
    device: &ash::Device,
    table: &GraphBindlessTable,
    slot: u32,
    view: vk::ImageView,
) -> Result<(), ComputeGraphError> {
    let storage_info = vk::DescriptorImageInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::GENERAL);
    let sampled_info = vk::DescriptorImageInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(table.set)
            .dst_binding(0)
            .dst_array_element(slot)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&storage_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(table.set)
            .dst_binding(1)
            .dst_array_element(slot)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&sampled_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
    Ok(())
}

fn add_graph_buffer(
    device: &ash::Device,
    table: &GraphBindlessTable,
    slot: u32,
    buffer: vk::Buffer,
) -> Result<(), ComputeGraphError> {
    let info = vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .offset(0)
        .range(vk::WHOLE_SIZE);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(table.set)
        .dst_binding(2)
        .dst_array_element(slot)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(std::slice::from_ref(&info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
    Ok(())
}

fn create_graph_image(
    device: &ash::Device,
    image_info: &vk::ImageCreateInfo<'_>,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
) -> Result<(vk::Image, vk::DeviceMemory), ComputeGraphError> {
    let image = unsafe { device.create_image(image_info, None)? };
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let (memory_type, _) = find_memory_type(
        memory_properties,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
    )
    .or_else(|_| {
        find_memory_type(
            memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::empty(),
            vk::MemoryPropertyFlags::empty(),
        )
    })
    .map_err(|error| ComputeGraphError::Invalid(error.to_string()))?;
    let allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    let memory = unsafe { device.allocate_memory(&allocate, None)? };
    if let Err(error) = unsafe { device.bind_image_memory(image, memory, 0) } {
        unsafe {
            device.free_memory(memory, None);
            device.destroy_image(image, None);
        }
        return Err(error.into());
    }
    Ok((image, memory))
}
