#include "../../shaders/compute_graph.hlsl"

[numthreads(8, 8, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    bindless_images[compute_graph.slots[1]][dispatch_thread_id.xy] =
        bindless_textures[compute_graph.slots[0]].Load(int3(dispatch_thread_id.xy, 0));
}
