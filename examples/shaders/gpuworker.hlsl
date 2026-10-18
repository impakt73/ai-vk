#include "../../shaders/compute_graph.hlsl"

[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint element_count;
    uint element_stride;
    bindless_buffers[compute_graph.slots[0]].GetDimensions(element_count, element_stride);
    if (dispatch_thread_id.x >= element_count)
    {
        return;
    }

    bindless_buffers[compute_graph.slots[0]][dispatch_thread_id.x] = 0;
}
