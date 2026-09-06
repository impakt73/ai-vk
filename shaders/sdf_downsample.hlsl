#include "compute_graph.hlsl"

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

    float2 uv = (float2(dispatch_thread_id.xy) + 0.5) / float2(width, height);
    float4 color = bindless_textures[
        compute_graph.slots[0]].SampleLevel(bindless_sampler, uv, 0.0);
    bindless_images[compute_graph.slots[1]][dispatch_thread_id.xy] = color;
}
