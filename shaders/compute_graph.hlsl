// Include this file from graph shaders. The table is rewritten before every
// dispatch and its slots index the persistent bindless resource tables.
RWTexture2D<float4> bindless_images[64] : register(u0, space1);
Texture2D<float4> bindless_textures[64] : register(t1, space1);
RWStructuredBuffer<uint> bindless_buffers[64] : register(u2, space1);
SamplerState bindless_sampler : register(s0, space0);

struct ComputeGraphPushConstants
{
    uint slots[16];
};

[[vk::push_constant]]
ComputeGraphPushConstants compute_graph;
