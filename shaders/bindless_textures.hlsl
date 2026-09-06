// Sampled images use a parallel descriptor set to the storage-image table.
Texture2D<float4> bindless_textures[64] : register(t1, space1);
SamplerState bindless_sampler : register(s0, space0);
