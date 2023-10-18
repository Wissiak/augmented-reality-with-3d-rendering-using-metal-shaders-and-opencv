/*
CPP translation of original Objective-C MetalAdder.m. Some stuff has been moved over to
the header. Source: https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu?language=objc

Original distribution license: LICENSE-original.txt.

Abstract:
A class to manage all of the Metal objects this app creates.
*/

#include "MetalAdder.hpp"
#include <iostream>

MetalAdder::MetalAdder(MTL::Device *device)
{

    _mDevice = device;

    NS::Error *error = nullptr;

    // Load the shader files with a .metal file extension in the project
    NS::String* filePath = NS::String::string("add.metallib", NS::UTF8StringEncoding);
    MTL::Library *defaultLibrary = _mDevice->newLibrary(filePath, &error);

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
    _mBufferResult = _mDevice->newBuffer(bufferSize, MTL::ResourceStorageModeShared);

    prepareData();
}

void MetalAdder::prepareData()
{

    generateRandomFloatData(_mBufferA);
    generateRandomFloatData(_mBufferB);
}

void MetalAdder::sendComputeCommand()
{
    // Create a command buffer to hold commands.
    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);

    // Start a compute pass.
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    encodeAddCommand(computeEncoder);

    // End the compute pass.
    computeEncoder->endEncoding();

    // Execute the command.
    commandBuffer->commit();

    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    commandBuffer->waitUntilCompleted();
}

void MetalAdder::encodeAddCommand(MTL::ComputeCommandEncoder *computeEncoder)
{
    // Encode the pipeline state object and its parameters.
    computeEncoder->setComputePipelineState(_mAddFunctionPSO);
    computeEncoder->setBuffer(_mBufferA, 0, 0);
    computeEncoder->setBuffer(_mBufferB, 0, 1);
    computeEncoder->setBuffer(_mBufferResult, 0, 2);

    MTL::Size gridSize = MTL::Size::Make(arrayLength, 1, 1);

    // Calculate a threadgroup size.
    NS::UInteger threadGroupSize = _mAddFunctionPSO->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > arrayLength)
    {
        threadGroupSize = arrayLength;
    }
    MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);

    // Encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
}

void MetalAdder::generateRandomFloatData(MTL::Buffer *buffer)
{
    // The pointer needs to be explicitly cast in C++, a difference from
    // Objective-C.
    float *dataPtr = (float *)buffer->contents();

    for (unsigned long index = 0; index < arrayLength; index++)
    {
        dataPtr[index] = (float)rand() / (float)(RAND_MAX);
    }
}

void MetalAdder::verifyResults()
{
    float *a = (float *)_mBufferA->contents();
    float *b = (float *)_mBufferB->contents();
    float *result = (float *)_mBufferResult->contents();

    for (int i = 0; i < arrayLength; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << result[i] << std::endl;
    } 

    for (unsigned long index = 0; index < arrayLength; index++)
    {
        if (result[index] != (a[index] + b[index]))
        {
            printf("Compute ERROR: index=%lu result=%g vs %g + %g = %g\n",
                   index, result[index], a[index], b[index], a[index] + b[index]);
            assert(result[index] == (a[index] + b[index]));
        }
    }
}

MetalAdder::~MetalAdder()
{
    _mBufferA->release();
    _mBufferB->release();
    _mBufferResult->release();

    _mAddFunctionPSO->release();
    _mCommandQueue->release();
}