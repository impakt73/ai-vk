#include "../../shaders/compute_graph.hlsl"

static const uint triangle_header_words = 1;
static const uint triangle_words = 10;

struct RasterTriangle
{
    float2 vertices[3];
    float4 color;
};

float triangle_word(uint word)
{
    return asfloat(bindless_buffers[compute_graph.slots[0]][word]);
}

RasterTriangle load_triangle(uint index)
{
    uint base = triangle_header_words + index * triangle_words;
    RasterTriangle tri;
    tri.vertices[0] = float2(triangle_word(base + 0), triangle_word(base + 1));
    tri.vertices[1] = float2(triangle_word(base + 2), triangle_word(base + 3));
    tri.vertices[2] = float2(triangle_word(base + 4), triangle_word(base + 5));
    tri.color = float4(
        triangle_word(base + 6),
        triangle_word(base + 7),
        triangle_word(base + 8),
        triangle_word(base + 9));
    return tri;
}

float edge_function(float2 a, float2 b, float2 sample_point)
{
    return (sample_point.x - a.x) * (b.y - a.y) - (sample_point.y - a.y) * (b.x - a.x);
}

[numthreads(8, 8, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint width;
    uint height;
    bindless_images[compute_graph.slots[1]].GetDimensions(width, height);
    if (dispatch_thread_id.x >= width || dispatch_thread_id.y >= height)
    {
        return;
    }

    uint triangle_count = bindless_buffers[compute_graph.slots[0]][0];
    float2 resolution = float2(width, height);
    float2 pixel = float2(dispatch_thread_id.xy) + 0.5;
    float4 color = float4(0.035, 0.045, 0.07, 1.0);

    [loop]
    for (uint index = 0; index < triangle_count; ++index)
    {
        RasterTriangle tri = load_triangle(index);
        float2 v0 = tri.vertices[0] * resolution;
        float2 v1 = tri.vertices[1] * resolution;
        float2 v2 = tri.vertices[2] * resolution;

        float area = edge_function(v0, v1, v2);
        if (abs(area) < 1e-6)
        {
            continue;
        }

        float e0 = edge_function(v1, v2, pixel);
        float e1 = edge_function(v2, v0, pixel);
        float e2 = edge_function(v0, v1, pixel);
        if (area < 0.0)
        {
            e0 = -e0;
            e1 = -e1;
            e2 = -e2;
        }

        if (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0)
        {
            color = tri.color;
        }
    }

    bindless_images[compute_graph.slots[1]][dispatch_thread_id.xy] = color;
}
