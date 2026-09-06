#include "compute_graph.hlsl"

[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    bindless_buffers[compute_graph.slots[0]][dispatch_thread_id.x] = dispatch_thread_id.x * 3 + 7;
}
