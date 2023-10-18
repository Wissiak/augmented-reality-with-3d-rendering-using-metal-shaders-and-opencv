#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include "MetalEngine.hpp"
#include <iostream>

/*
MTLEngine() {

    _mDevice = device;

    NS::Error *error = nullptr;

    // Load the shader files with a .metal file extension in the project
    NS::String* filePath = NS::String::string("../add.metallib",
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

void MTLEngine::init() {
  initDevice();

  //metalLayer = new CA::MetalLayer();

  createCommandQueue();
  createDefaultLibrary();
  loadMeshes();
  createRenderPipeline();
  //createDepthAndMSAATextures();
  //createRenderPassDescriptor();
}

void MTLEngine::initDevice() { metalDevice = MTL::CreateSystemDefaultDevice(); }
void MTLEngine::createCommandQueue() {
  metalCommandQueue = metalDevice->newCommandQueue();
  assert(metalCommandQueue != nullptr);
}
void MTLEngine::createDefaultLibrary() {
  NS::String *filePath =
      NS::String::string("../model.metallib", NS::UTF8StringEncoding);
  metalDefaultLibrary = metalDevice->newLibrary(filePath, &error);

  assert(metalDefaultLibrary != nullptr);
}

void MTLEngine::loadMeshes() {
  model = new Model("../assets/tutorial.obj", metalDevice);

  std::cout << "Mesh Count: " << model->meshes.size() << std::endl;
}


void MTLEngine::createRenderPipeline() {
    MTL::Function* vertexShader = metalDefaultLibrary->newFunction(NS::String::string("vertexShader", NS::ASCIIStringEncoding));
    assert(vertexShader);
    MTL::Function* fragmentShader = metalDefaultLibrary->newFunction(NS::String::string("fragmentShader", NS::ASCIIStringEncoding));
    assert(fragmentShader);
    
    MTL::RenderPipelineDescriptor* renderPipelineDescriptor = MTL::RenderPipelineDescriptor::alloc()->init();
    renderPipelineDescriptor->setVertexFunction(vertexShader);
    renderPipelineDescriptor->setFragmentFunction(fragmentShader);
    assert(renderPipelineDescriptor);
    //MTL::PixelFormat pixelFormat = (MTL::PixelFormat)metalLayer->pixelFormat();
    //renderPipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(pixelFormat);
    renderPipelineDescriptor->setSampleCount(4);
    renderPipelineDescriptor->setLabel(NS::String::string("Model Render Pipeline", NS::ASCIIStringEncoding));
    renderPipelineDescriptor->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
    renderPipelineDescriptor->setTessellationOutputWindingOrder(MTL::WindingCounterClockwise);
    
    NS::Error* error;
    metalRenderPSO = metalDevice->newRenderPipelineState(renderPipelineDescriptor, &error);
    
    if (metalRenderPSO == nil) {
        std::cout << "Error creating render pipeline state: " << error << std::endl;
        std::exit(0);
    }
    
    MTL::DepthStencilDescriptor* depthStencilDescriptor = MTL::DepthStencilDescriptor::alloc()->init();
    depthStencilDescriptor->setDepthCompareFunction(MTL::CompareFunctionLessEqual);
    depthStencilDescriptor->setDepthWriteEnabled(true);
    depthStencilState = metalDevice->newDepthStencilState(depthStencilDescriptor);
    
    depthStencilDescriptor->release();
    renderPipelineDescriptor->release();
    vertexShader->release();
    fragmentShader->release();
}


void MTLEngine::run() {
}

void MTLEngine::cleanup() {
//    delete mesh;
    delete model;
    depthTexture->release();
    renderPassDescriptor->release();
    metalDevice->release();
}