#include "../../shaders/compute_graph.hlsl"

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

    uint square = dispatch_thread_id.x / 4 + dispatch_thread_id.y / 4;
    float value = (square & 1) == 0 ? 0.0 : 1.0;
    bindless_images[compute_graph.slots[0]][dispatch_thread_id.xy] = float4(value, value, value, 1.0);
}
