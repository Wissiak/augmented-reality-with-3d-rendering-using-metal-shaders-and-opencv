#include <iostream>
#include <filesystem>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

#include "model.hpp"
#include "MetalAdder.hpp"

int main(int argc, char *argv[]) {
    std::cout << "Current path is " << std::filesystem::current_path() << '\n';
    // Create GPU code / arrays --------------------------------------------------------
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    MetalAdder *adder = new MetalAdder(device);

    Model *m = new Model("assets/tutorial.obj", device);

    // Verify Metal code ---------------------------------------------------------------
    adder->sendComputeCommand(); // This computes the array sum
    adder->verifyResults();

    // Profile Metal code --------------------------------------------------------------
    adder->sendComputeCommand();

    // Verify serial code --------------------------------------------------------------
    // Get buffers pointers for CPU code. Using MTL::ResourceStorageModeShared should
    // make them accessible to both GPU and CPU, perfect!
    auto array_a = ((float *)adder->_mBufferA->contents());
    auto array_b = ((float *)adder->_mBufferB->contents());
    auto array_c = ((float *)adder->_mBufferResult->contents());

    // Let's randomize the data again, making sure that the result buffer starts out
    // incorrect
    adder->prepareData();

    adder->verifyResults();

    // Profile serial code -------------------------------------------------------------
       
    device->release();
}

