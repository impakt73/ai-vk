#include "compute_graph.hlsl"

[numthreads(8, 8, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    bindless_images[compute_graph.slots[0]][dispatch_thread_id.xy] = float4(0.047, 0.384, 0.788, 1.0);
}
