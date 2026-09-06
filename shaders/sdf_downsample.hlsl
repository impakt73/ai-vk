#include "bindless_images.hlsl"
#include "bindless_textures.hlsl"

struct PushConstants
{
    uint4 target_extent_and_source;
};

[[vk::push_constant]]
PushConstants push_constants;

[numthreads(8, 8, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint width = push_constants.target_extent_and_source.y;
    uint height = push_constants.target_extent_and_source.z;
    if (dispatch_thread_id.x >= width || dispatch_thread_id.y >= height)
    {
        return;
    }

    float2 uv = (float2(dispatch_thread_id.xy) + 0.5) / float2(width, height);
    float4 color = bindless_textures[
        push_constants.target_extent_and_source.w].SampleLevel(bindless_sampler, uv, 0.0);
    bindless_images[push_constants.target_extent_and_source.x][dispatch_thread_id.xy] = color;
}
