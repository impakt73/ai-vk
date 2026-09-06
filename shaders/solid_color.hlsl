#include "bindless_images.hlsl"

struct PushConstants
{
    uint4 target_and_extent;
    float4 color;
};

[[vk::push_constant]]
PushConstants push_constants;

[numthreads(8, 8, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    if (dispatch_thread_id.x >= push_constants.target_and_extent.y ||
        dispatch_thread_id.y >= push_constants.target_and_extent.z)
    {
        return;
    }

    bindless_images[push_constants.target_and_extent.x][dispatch_thread_id.xy] =
        push_constants.color;
}
