#pragma once

#include <simd/simd.h>

#include "../built-libs/aapl/AAPLMathUtilities.h"
#include "mesh.hpp"
#include "model.hpp"
#include <QuartzCore/CAMetalLayer.h>
#include <QuartzCore/CAMetalLayer.hpp>

#include <iostream>

class MTLEngine {
public:
  CA::MetalDrawable *run(simd_float4 lightPosition, float pitch, float yaw,
                         matrix_float4x4 modelMatrix);
  void init(int width, int height);
  void cleanup();

private:
  void initDevice();

  void createCommandQueue();
  void loadMeshes();
  void createDefaultLibrary();
  void createRenderPipeline();
  void createDepthAndMSAATextures();
  void createRenderPassDescriptor();

  // Upon resizing, update Depth and MSAA Textures.
  void updateRenderPassDescriptor();

  void draw();
  void sendRenderCommand(simd_float4 lightPosition, float pitch, float yaw,
                         matrix_float4x4 modelMatrix);
  void encodeRenderCommand(MTL::RenderCommandEncoder *renderCommandEncoder,
                           simd_float4 lightPosition, float pitch, float yaw,
                           matrix_float4x4 modelMatrix);

  MTL::Device *metalDevice;
  CA::MetalLayer *metalLayer;
  CA::MetalDrawable *metalDrawable;

  float fov = 45.0f * (M_PI / 180.0f);
  float nearZ = 0.1f;
  float farZ = 10000.0f;

  float3 position = make_float3(0, 0, 0);
  float3 up = make_float3(0, -1, 0);
  float3 front = make_float3(0, 0, 1);
  simd_float4 lightColor = simd_make_float4(1.0, 1.0, 1.0, 0.5);

  NS::Error *error = nullptr;
  MTL::Library *metalDefaultLibrary;
  MTL::CommandQueue *metalCommandQueue;
  MTL::CommandBuffer *metalCommandBuffer;
  MTL::RenderPipelineState *metalRenderPSO;
  MTL::RenderPassDescriptor *renderPassDescriptor;
  Mesh *mesh;
  MTL::DepthStencilState *depthStencilState;
  MTL::Texture *msaaRenderTargetTexture;
  MTL::Texture *depthTexture;
  Model *model;
  int sampleCount = 4;
};
