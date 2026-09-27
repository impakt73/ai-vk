#include "../../shaders/compute_graph.hlsl"

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

    float4 color = float4(0.0, 0.0, 0.0, 1.0);
    if (compute_graph.frame_index > 0)
    {
        color = bindless_textures[compute_graph.slots[0]].Load(
            int3(dispatch_thread_id.xy, 0));
    }
    color.rgb += 0.25;
    bindless_images[compute_graph.slots[1]][dispatch_thread_id.xy] = color;
}
