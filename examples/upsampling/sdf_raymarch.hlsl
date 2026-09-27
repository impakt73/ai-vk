#include "../../shaders/compute_graph.hlsl"

static const float far_plane = 16.0;
static const float sphere_radius = 0.82;
static const float motion_display_scale = 24.0;

float sphere_sdf(float3 position, float3 center, float radius)
{
    return length(position - center) - radius;
}

float3 sphere_center(uint index, float time)
{
    float phase = time + float(index) * 2.0943951;
    return float3(
        (float(index) - 1.0) * 1.15 + 0.12 * sin(phase),
        0.28 * sin(phase * 1.2),
        0.35 * cos(phase * 0.8));
}

float plane_sdf(float3 position)
{
    return position.z + 4.0;
}

float scene_sdf(float3 position, float time)
{
    float distance = plane_sdf(position);
    for (uint index = 0; index < 3; ++index)
    {
        distance = min(distance, sphere_sdf(position, sphere_center(index, time), sphere_radius));
    }
    return distance;
}

int scene_material(float3 position, float time)
{
    int material = 3;
    float closest = plane_sdf(position);
    for (uint index = 0; index < 3; ++index)
    {
        float distance = sphere_sdf(position, sphere_center(index, time), sphere_radius);
        if (distance < closest)
        {
            closest = distance;
            material = int(index);
        }
    }
    return material;
}

float3 camera_position(float time)
{
    return float3(
        0.8 * sin(time * 1.7),
        0.15 + 0.3 * sin(time * 2.3),
        4.6 + 0.5 * cos(time * 1.7));
}

float3 ray_direction(float2 uv, float aspect)
{
    return normalize(float3(uv.x * aspect, -uv.y, -2.2));
}

float2 project_to_pixels(float3 position, float3 camera, float2 resolution)
{
    float3 offset = position - camera;
    float aspect = resolution.x / resolution.y;
    float2 uv;
    uv.x = -2.2 * offset.x / offset.z / aspect;
    uv.y = 2.2 * offset.y / offset.z;
    return (uv * 0.5 + 0.5) * resolution;
}

bool raymarch(float3 origin, float3 direction, float time, out float distance)
{
    distance = 0.0;
    [loop]
    for (int step = 0; step < 96; ++step)
    {
        float scene_distance = scene_sdf(origin + direction * distance, time);
        if (scene_distance < 0.001)
        {
            return true;
        }
        distance += scene_distance;
        if (distance > 20.0)
        {
            break;
        }
    }
    return false;
}

float3 scene_normal(float3 position, float time)
{
    const float epsilon = 0.001;
    const float3 x = float3(epsilon, 0.0, 0.0);
    const float3 y = float3(0.0, epsilon, 0.0);
    const float3 z = float3(0.0, 0.0, epsilon);
    return normalize(float3(
        scene_sdf(position + x, time) - scene_sdf(position - x, time),
        scene_sdf(position + y, time) - scene_sdf(position - y, time),
        scene_sdf(position + z, time) - scene_sdf(position - z, time)));
}

float3 phong(float3 position, float3 normal, float3 ray_direction, int material)
{
    float3 base_color;
    if (material == 3)
    {
        float checker = frac((floor(position.x * 0.5) + floor(position.y * 0.5)) * 0.5) * 2.0;
        base_color = lerp(float3(0.16, 0.18, 0.22), float3(0.34, 0.36, 0.40), checker);
    }
    else
    {
        const float3 material_colors[3] = {
            float3(0.85, 0.18, 0.12),
            float3(0.12, 0.42, 0.9),
            float3(0.95, 0.62, 0.12)
        };
        base_color = material_colors[material];
    }
    float3 light_position = float3(-3.5, 4.5, 4.0);
    float3 light_direction = normalize(light_position - position);
    float diffuse = max(dot(normal, light_direction), 0.0);
    float3 reflected = reflect(-light_direction, normal);
    float specular = pow(max(dot(reflected, -ray_direction), 0.0), 48.0);
    return base_color * (0.12 + 0.82 * diffuse) + float3(1.0, 1.0, 1.0) * (0.35 * specular);
}

[numthreads(8, 8, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    float current_time = asfloat(bindless_buffers[compute_graph.slots[0]][0]);
    float previous_time = asfloat(bindless_buffers[compute_graph.slots[0]][1]);
    uint width;
    uint height;
    bindless_images[compute_graph.slots[1]].GetDimensions(width, height);
    if (dispatch_thread_id.x >= width || dispatch_thread_id.y >= height)
    {
        return;
    }

    float2 resolution = float2(width, height);
    float2 pixel = float2(dispatch_thread_id.xy) + 0.5;
    float2 uv = pixel / resolution * 2.0 - 1.0;

    float3 current_camera = camera_position(current_time);
    float3 previous_camera = camera_position(previous_time);
    float3 direction = ray_direction(uv, resolution.x / resolution.y);

    float distance;
    float3 color = float3(0.015, 0.02, 0.04) + float3(0.02, 0.03, 0.06) * (1.0 - uv.y);
    float3 normal = float3(0.0, 0.0, 1.0);
    float depth = far_plane;
    float2 motion = float2(0.0, 0.0);
    if (raymarch(current_camera, direction, current_time, distance))
    {
        float3 position = current_camera + direction * distance;
        int material = scene_material(position, current_time);
        normal = scene_normal(position, current_time);
        color = phong(position, normal, direction, material);
        depth = distance;

        float3 previous_position = position;
        if (material < 3)
        {
            uint sphere_index = uint(material);
            previous_position +=
                sphere_center(sphere_index, previous_time) - sphere_center(sphere_index, current_time);
        }
        if (previous_position.z < previous_camera.z)
        {
            float2 previous_pixel = project_to_pixels(previous_position, previous_camera, resolution);
            motion = (previous_pixel - pixel) / resolution;
        }
    }

    color = pow(saturate(color), 1.0 / 2.2);
    float depth_encoded = saturate(depth / far_plane);
    float2 motion_encoded = saturate(motion * motion_display_scale + 0.5);

    bindless_images[compute_graph.slots[1]][dispatch_thread_id.xy] = float4(color, 1.0);
    bindless_images[compute_graph.slots[2]][dispatch_thread_id.xy] = float4(normal * 0.5 + 0.5, 1.0);
    bindless_images[compute_graph.slots[3]][dispatch_thread_id.xy] = float4(motion_encoded, 0.0, 1.0);
    bindless_images[compute_graph.slots[4]][dispatch_thread_id.xy] = float4(depth_encoded.xxx, 1.0);
}
