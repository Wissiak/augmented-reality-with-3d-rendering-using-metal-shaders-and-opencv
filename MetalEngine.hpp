//
//  mtl_engine.hpp
//  MetalTutorial
//

#pragma once

#include <Metal/Metal.hpp>
#include <Metal/Metal.h>
#include <QuartzCore/CAMetalLayer.hpp>
#include <QuartzCore/CAMetalLayer.h>
#include <QuartzCore/QuartzCore.hpp>
#include <simd/simd.h>

#include "AAPLMathUtilities.h"
#include "mesh.hpp"
#include "model.hpp"
#include "camera.hpp"

#include <iostream>
#include <filesystem>

class MTLEngine {
public:
    void init();
    void run();
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
    void sendRenderCommand();
    void encodeRenderCommand(MTL::RenderCommandEncoder* renderCommandEncoder);
    
    MTL::Device* metalDevice;
    CA::MetalDrawable* metalDrawable;
    bool windowResizeFlag = false;
    int newWidth, newHeight;
    
    MTL::Library* metalDefaultLibrary;
    MTL::CommandQueue* metalCommandQueue;
    MTL::CommandBuffer* metalCommandBuffer;
    MTL::RenderPipelineState* metalRenderPSO;
    MTL::RenderPassDescriptor* renderPassDescriptor;
    Mesh* mesh;
    MTL::DepthStencilState* depthStencilState;
    MTL::Texture* depthTexture;
    Model* model;
    int sampleCount = 4;
    
    // Camera
    Camera camera;
    float lastX;
    float lastY;
    bool firstMouse = true;

    // Timing
    float deltaTime = 0.0f;    // time between current frame and last frame
    float lastFrame = 0.0f;
    
};
