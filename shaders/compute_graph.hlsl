// Include this file from graph shaders. The table is rewritten before every
// dispatch and its slots index the persistent bindless resource tables.
#include "bindless_images.hlsl"
#include "bindless_textures.hlsl"

RWStructuredBuffer<uint> bindless_buffers[64] : register(u2, space1);

struct ComputeGraphPushConstants
{
    uint slots[30];
};

[[vk::push_constant]]
ComputeGraphPushConstants compute_graph;
