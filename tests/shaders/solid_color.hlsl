#include "../../shaders/compute_graph.hlsl"

#ifndef AI_VK_SOLID_COLOR
#define AI_VK_SOLID_COLOR float4(12.0 / 255.0, 98.0 / 255.0, 201.0 / 255.0, 1.0)
#endif

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

    bindless_images[compute_graph.slots[0]][dispatch_thread_id.xy] =
        AI_VK_SOLID_COLOR;
}
