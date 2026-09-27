#include "../../shaders/compute_graph.hlsl"

static const float motion_display_scale = 24.0;

[numthreads(8, 8, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint width;
    uint height;
    bindless_images[compute_graph.slots[5]].GetDimensions(width, height);
    if (dispatch_thread_id.x >= width || dispatch_thread_id.y >= height)
    {
        return;
    }

    float2 uv = (float2(dispatch_thread_id.xy) + 0.5) / float2(width, height);
    float4 color = bindless_textures[
        compute_graph.slots[0]].SampleLevel(bindless_sampler, uv, 0.0);
    float4 normal = bindless_textures[
        compute_graph.slots[1]].SampleLevel(bindless_sampler, uv, 0.0);
    float4 motion = bindless_textures[
        compute_graph.slots[2]].SampleLevel(bindless_sampler, uv, 0.0);
    float4 depth = bindless_textures[
        compute_graph.slots[3]].SampleLevel(bindless_sampler, uv, 0.0);

    float2 motion_vector = (motion.xy - 0.5) / motion_display_scale;
    float2 previous_uv = uv + motion_vector;
    bool valid = compute_graph.frame_index > 0
        && all(previous_uv >= 0.0)
        && all(previous_uv < 1.0);
    if (valid)
    {
        float previous_depth = bindless_textures[
            compute_graph.slots[4]].SampleLevel(bindless_sampler, previous_uv, 0.0).x;
        valid = abs(depth.x - previous_depth) <= compute_graph.parameter;
    }

    bindless_images[compute_graph.slots[5]][dispatch_thread_id.xy] = color;
    bindless_images[compute_graph.slots[6]][dispatch_thread_id.xy] = normal;
    bindless_images[compute_graph.slots[7]][dispatch_thread_id.xy] = motion;
    bindless_images[compute_graph.slots[8]][dispatch_thread_id.xy] = depth;
    bindless_images[compute_graph.slots[9]][dispatch_thread_id.xy] = valid
        ? float4(0.0, 1.0, 0.0, 1.0)
        : float4(1.0, 0.0, 0.0, 1.0);
}
