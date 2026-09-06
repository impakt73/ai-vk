// Keep this declaration in one include so every shader uses the same table.
RWTexture2D<float4> bindless_images[64] : register(u0, space0);
