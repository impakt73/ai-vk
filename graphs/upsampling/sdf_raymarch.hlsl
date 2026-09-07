#include "../../shaders/compute_graph.hlsl"

float sphere_sdf(float3 position, float3 center, float radius)
{
    return length(position - center) - radius;
}

float scene_sdf(float3 position)
{
    float first = sphere_sdf(position, float3(-1.15, 0.05, 0.0), 0.82);
    float second = sphere_sdf(position, float3(0.0, 0.15, -0.65), 0.82);
    float third = sphere_sdf(position, float3(1.15, -0.05, 0.0), 0.82);
    return min(first, min(second, third));
}

int scene_material(float3 position)
{
    float first = sphere_sdf(position, float3(-1.15, 0.05, 0.0), 0.82);
    float second = sphere_sdf(position, float3(0.0, 0.15, -0.65), 0.82);
    return first < second && first < sphere_sdf(position, float3(1.15, -0.05, 0.0), 0.82)
        ? 0
        : (second < sphere_sdf(position, float3(1.15, -0.05, 0.0), 0.82) ? 1 : 2);
}

bool raymarch(float3 origin, float3 direction, out float distance)
{
    distance = 0.0;
    [loop]
    for (int step = 0; step < 96; ++step)
    {
        float scene_distance = scene_sdf(origin + direction * distance);
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

float3 scene_normal(float3 position)
{
    const float epsilon = 0.001;
    const float3 x = float3(epsilon, 0.0, 0.0);
    const float3 y = float3(0.0, epsilon, 0.0);
    const float3 z = float3(0.0, 0.0, epsilon);
    return normalize(float3(
        scene_sdf(position + x) - scene_sdf(position - x),
        scene_sdf(position + y) - scene_sdf(position - y),
        scene_sdf(position + z) - scene_sdf(position - z)));
}

float3 phong(float3 position, float3 normal, float3 ray_direction, int material)
{
    const float3 material_colors[3] = {
        float3(0.85, 0.18, 0.12),
        float3(0.12, 0.42, 0.9),
        float3(0.95, 0.62, 0.12)
    };
    float3 base_color = material_colors[material];
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
    uint width;
    uint height;
    bindless_images[compute_graph.slots[0]].GetDimensions(width, height);
    if (dispatch_thread_id.x >= width || dispatch_thread_id.y >= height)
    {
        return;
    }

    float2 pixel = float2(dispatch_thread_id.xy) + 0.5;
    float2 uv = pixel / float2(width, height) * 2.0 - 1.0;
    uv.x *= float(width) / float(height);

    float3 camera = float3(0.0, 0.1, 4.6);
    float3 ray_direction = normalize(float3(uv.x, -uv.y, -2.2));
    float distance;
    float3 color = float3(0.015, 0.02, 0.04) + float3(0.02, 0.03, 0.06) * (1.0 - uv.y);
    if (raymarch(camera, ray_direction, distance))
    {
        float3 position = camera + ray_direction * distance;
        float3 normal = scene_normal(position);
        color = phong(position, normal, ray_direction, scene_material(position));
    }

    color = pow(saturate(color), 1.0 / 2.2);
    bindless_images[compute_graph.slots[0]][dispatch_thread_id.xy] = float4(color, 1.0);
}
