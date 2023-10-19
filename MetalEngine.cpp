#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "MetalEngine.hpp"
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.h>
#include <QuartzCore/QuartzCore.hpp>
#include <iostream>

/*
MTLEngine() {

    _mDevice = device;

    NS::Error *error = nullptr;

    // Load the shader files with a .metal file extension in the project
    NS::String* filePath = NS::String::string("./add.metallib",
NS::UTF8StringEncoding); MTL::Library *defaultLibrary =
_mDevice->newLibrary(filePath, &error);

    assert(defaultLibrary != nullptr);


    auto str = NS::String::string("add_arrays", NS::ASCIIStringEncoding);
    MTL::Function *addFunction = defaultLibrary->newFunction(str);
    defaultLibrary->release();

    assert(addFunction != nullptr);

    // Create a compute pipeline state object.
    _mAddFunctionPSO = _mDevice->newComputePipelineState(addFunction, &error);
    addFunction->release();

    assert(_mAddFunctionPSO != nullptr);

    _mCommandQueue = _mDevice->newCommandQueue();

    assert(_mCommandQueue != nullptr);

    // Allocate three buffers to hold our initial data and the result.
    _mBufferA = _mDevice->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
    _mBufferB = _mDevice->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
    _mBufferResult = _mDevice->newBuffer(bufferSize,
MTL::ResourceStorageModeShared);

    prepareData();
}*/

void MTLEngine::init(int width, int height) {
  initDevice();

  metalLayer = CA::MetalLayer::layer();
  metalLayer->setDevice(metalDevice);
  metalLayer->setPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm);
  metalLayer->setDrawableSize(CGSizeMake(width, height));

  metalDrawable = metalLayer->nextDrawable();

  createCommandQueue();
  createDefaultLibrary();
  loadMeshes();
  createRenderPipeline();
  createDepthAndMSAATextures();
  createRenderPassDescriptor();
}

void MTLEngine::initDevice() { metalDevice = MTL::CreateSystemDefaultDevice(); }
void MTLEngine::createCommandQueue() {
  metalCommandQueue = metalDevice->newCommandQueue();
  assert(metalCommandQueue != nullptr);
}
void MTLEngine::createDefaultLibrary() {
  NS::String *filePath =
      NS::String::string("./build/model.metallib", NS::UTF8StringEncoding);
  metalDefaultLibrary = metalDevice->newLibrary(filePath, &error);

  assert(metalDefaultLibrary != nullptr);
}

void MTLEngine::loadMeshes() {
  model = new Model("./assets/tutorial.obj", metalDevice);

  std::cout << "Mesh Count: " << model->meshes.size() << std::endl;
}

void MTLEngine::createRenderPipeline() {
  MTL::Function *vertexShader = metalDefaultLibrary->newFunction(
      NS::String::string("vertexShader", NS::ASCIIStringEncoding));
  assert(vertexShader);
  MTL::Function *fragmentShader = metalDefaultLibrary->newFunction(
      NS::String::string("fragmentShader", NS::ASCIIStringEncoding));
  assert(fragmentShader);

  MTL::RenderPipelineDescriptor *renderPipelineDescriptor =
      MTL::RenderPipelineDescriptor::alloc()->init();
  renderPipelineDescriptor->setVertexFunction(vertexShader);
  renderPipelineDescriptor->setFragmentFunction(fragmentShader);
  assert(renderPipelineDescriptor);
  MTL::PixelFormat pixelFormat = (MTL::PixelFormat)metalLayer->pixelFormat();
  renderPipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(
      pixelFormat);
  renderPipelineDescriptor->setSampleCount(4);
  renderPipelineDescriptor->setLabel(
      NS::String::string("Model Render Pipeline", NS::ASCIIStringEncoding));
  renderPipelineDescriptor->setDepthAttachmentPixelFormat(
      MTL::PixelFormatDepth32Float);
  renderPipelineDescriptor->setTessellationOutputWindingOrder(
      MTL::WindingCounterClockwise);

  NS::Error *error;
  metalRenderPSO =
      metalDevice->newRenderPipelineState(renderPipelineDescriptor, &error);

  if (metalRenderPSO == nil) {
    std::cout << "Error creating render pipeline state: " << error << std::endl;
    std::exit(0);
  }

  MTL::DepthStencilDescriptor *depthStencilDescriptor =
      MTL::DepthStencilDescriptor::alloc()->init();
  depthStencilDescriptor->setDepthCompareFunction(
      MTL::CompareFunctionLessEqual);
  depthStencilDescriptor->setDepthWriteEnabled(true);
  depthStencilState = metalDevice->newDepthStencilState(depthStencilDescriptor);

  depthStencilDescriptor->release();
  renderPipelineDescriptor->release();
  vertexShader->release();
  fragmentShader->release();
}

void MTLEngine::updateRenderPassDescriptor() {
  renderPassDescriptor->colorAttachments()->object(0)->setTexture(
      msaaRenderTargetTexture);
  renderPassDescriptor->colorAttachments()->object(0)->setResolveTexture(
      metalDrawable->texture());
  renderPassDescriptor->depthAttachment()->setTexture(depthTexture);
}

void MTLEngine::sendRenderCommand(float3 position, float pitch, float yaw, matrix_float4x4 rotationMatrix, matrix_float4x4 modelMatrix) {
  metalCommandBuffer = metalCommandQueue->commandBuffer();

  updateRenderPassDescriptor();
  MTL::RenderCommandEncoder *renderCommandEncoder =
      metalCommandBuffer->renderCommandEncoder(renderPassDescriptor);
  encodeRenderCommand(renderCommandEncoder, position, pitch, yaw, rotationMatrix, modelMatrix);
  renderCommandEncoder->endEncoding();

  metalCommandBuffer->presentDrawable(metalDrawable);
  metalCommandBuffer->commit();
  metalCommandBuffer->waitUntilCompleted();
}

float radians(float degrees) { return degrees * M_PI / 180.0f; }

simd::float4x4 lookAt(const simd::float3 eye, const simd::float3 center,
                      const simd::float3 up) {
  float3 z = normalize(eye - center);
  float3 x = normalize(simd::cross(up, z));
  float3 y = cross(z, x);

  simd::float4x4 viewMatrix;
  viewMatrix.columns[0] = make_float4(x, 0);
  viewMatrix.columns[1] = make_float4(y, 0);
  viewMatrix.columns[2] = make_float4(z, 0);
  viewMatrix.columns[3] = make_float4(eye, 1);

  return simd::inverse(viewMatrix);
}

void MTLEngine::encodeRenderCommand(
    MTL::RenderCommandEncoder *renderCommandEncoder, float3 position, float pitch, float yaw, matrix_float4x4 rotationMatrix, matrix_float4x4 modelMatrix) {
  renderCommandEncoder->setFrontFacingWinding(MTL::WindingCounterClockwise);
  renderCommandEncoder->setCullMode(MTL::CullModeBack);
  // If you want to render in wire-frame mode, you can uncomment this line!
  // renderCommandEncoder->setTriangleFillMode(MTL::TriangleFillModeLines);
  renderCommandEncoder->setRenderPipelineState(metalRenderPSO);
  renderCommandEncoder->setDepthStencilState(depthStencilState);

  // Aspect ratio should match the ratio between the window width and height,
  // otherwise the image will look stretched.
  float aspectRatio = (metalDrawable->layer()->drawableSize().width /
                       metalDrawable->layer()->drawableSize().height);
  float fov = 45.0f * (M_PI / 180.0f);
  float nearZ = 0.1f;
  float farZ = 10000.0f;
  matrix_float4x4 perspectiveMatrix =
      matrix_perspective_right_hand(fov, aspectRatio, nearZ, farZ);
  MTL::PrimitiveType typeTriangle = MTL::PrimitiveTypeTriangle;
  MTL::SamplerDescriptor *samplerDescriptor =
      MTL::SamplerDescriptor::alloc()->init();
  samplerDescriptor->setMinFilter(MTL::SamplerMinMagFilterLinear);
  samplerDescriptor->setMipFilter(MTL::SamplerMipFilterLinear);
  MTL::SamplerState *samplerState =
      metalDevice->newSamplerState(samplerDescriptor);

  float3 up = make_float3(0.0f, 1.0f, 0.0f);

  float3 front;
  front.x = cos(radians(yaw)) * cos(radians(pitch));
  front.y = sin(radians(pitch));
  front.z = sin(radians(yaw)) * cos(radians(pitch));
  front = normalize(front);
  // also re-calculate the Right and Up vector
  float3 right = normalize(cross(front, up));
  up = normalize(cross(right, front));

  matrix_float4x4 viewMatrix =
      lookAt(position, position + front, up);
  for (Mesh *mesh : model->meshes) {
    renderCommandEncoder->setVertexBuffer(mesh->vertexBuffer, 0, 0);
    renderCommandEncoder->setVertexBytes(&modelMatrix, sizeof(modelMatrix), 1);
    renderCommandEncoder->setVertexBytes(&viewMatrix, sizeof(viewMatrix), 2);
    renderCommandEncoder->setVertexBytes(&perspectiveMatrix,
                                         sizeof(perspectiveMatrix), 3);
    renderCommandEncoder->setFragmentBytes(&position, sizeof(position), 0);
    renderCommandEncoder->setFragmentTexture(model->textures->textureArray, 1);
    renderCommandEncoder->setFragmentBuffer(model->textures->textureInfosBuffer,
                                            0, 2);
    renderCommandEncoder->setFragmentBytes(&modelMatrix, sizeof(modelMatrix),
                                           3);
    renderCommandEncoder->setFragmentSamplerState(samplerState, 4);

    renderCommandEncoder->drawIndexedPrimitives(typeTriangle, mesh->indexCount,
                                                MTL::IndexTypeUInt32,
                                                mesh->indexBuffer, 0);
  }
}

CA::MetalDrawable *MTLEngine::run(float3 position, float pitch, float yaw, matrix_float4x4 rotationMatrix, matrix_float4x4 modelMatrix) {
  sendRenderCommand(position, pitch, yaw, rotationMatrix, modelMatrix);
  metalDrawable = metalLayer->nextDrawable();

  /*MTL::TextureDescriptor *desc = MTL::TextureDescriptor::alloc()->init();
  desc->setTextureType(MTL::TextureType::TextureType2D);
  desc->setPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm);
  desc->setWidth(drawable->texture()->width());
  desc->setHeight(drawable->texture()->height());

  MTL::Texture* texture = metalDevice->newTexture(desc);
  MTL::RenderPassDescriptor *renderPassDesc =
  MTL::RenderPassDescriptor::alloc()->init();
  MTL::RenderPassColorAttachmentDescriptor *colorAttachment =
      renderPassDescriptor->colorAttachments()->object(0);

  colorAttachment->setTexture(texture);
  colorAttachment->setLoadAction(MTL::LoadAction::LoadActionClear);
  colorAttachment->setClearColor({0.0, 0.0, 0.0, 1.0});

  MTL::CommandBuffer* commandBuffer = metalCommandQueue->commandBuffer();
  MTL::RenderCommandEncoder* encoder =
      commandBuffer->renderCommandEncoder(renderPassDesc);
  // Your rendering commands here
  encoder->endEncoding();
  commandBuffer->presentDrawable(drawable);
  commandBuffer->commit();*/
  return metalDrawable;
}

void MTLEngine::createDepthAndMSAATextures() {
  MTL::TextureDescriptor *msaaTextureDescriptor =
      MTL::TextureDescriptor::alloc()->init();
  msaaTextureDescriptor->setTextureType(MTL::TextureType2DMultisample);
  msaaTextureDescriptor->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
  msaaTextureDescriptor->setWidth(metalLayer->drawableSize().width);
  msaaTextureDescriptor->setHeight(metalLayer->drawableSize().height);
  msaaTextureDescriptor->setSampleCount(sampleCount);
  msaaTextureDescriptor->setUsage(MTL::TextureUsageRenderTarget);

  msaaRenderTargetTexture = metalDevice->newTexture(msaaTextureDescriptor);

  MTL::TextureDescriptor *depthTextureDescriptor =
      MTL::TextureDescriptor::alloc()->init();
  depthTextureDescriptor->setTextureType(MTL::TextureType2DMultisample);
  depthTextureDescriptor->setPixelFormat(MTL::PixelFormatDepth32Float);
  depthTextureDescriptor->setWidth(metalLayer->drawableSize().width);
  depthTextureDescriptor->setHeight(metalLayer->drawableSize().height);
  depthTextureDescriptor->setUsage(MTL::TextureUsageRenderTarget);
  depthTextureDescriptor->setSampleCount(sampleCount);

  depthTexture = metalDevice->newTexture(depthTextureDescriptor);

  msaaTextureDescriptor->release();
  depthTextureDescriptor->release();
}

void MTLEngine::createRenderPassDescriptor() {
  renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();

  MTL::RenderPassColorAttachmentDescriptor *colorAttachment =
      renderPassDescriptor->colorAttachments()->object(0);
  MTL::RenderPassDepthAttachmentDescriptor *depthAttachment =
      renderPassDescriptor->depthAttachment();

  colorAttachment->setTexture(msaaRenderTargetTexture);
  colorAttachment->setResolveTexture(metalDrawable->texture());
  colorAttachment->setLoadAction(MTL::LoadActionClear);
  colorAttachment->setClearColor(MTL::ClearColor(0.0, 0.0, 0.0, 0.0));
  colorAttachment->setStoreAction(MTL::StoreActionMultisampleResolve);

  depthAttachment->setTexture(depthTexture);
  depthAttachment->setLoadAction(MTL::LoadActionClear);
  depthAttachment->setStoreAction(MTL::StoreActionDontCare);
  depthAttachment->setClearDepth(1.0);
}

void MTLEngine::cleanup() {
  delete model;
  depthTexture->release();
  renderPassDescriptor->release();
  metalDevice->release();
}