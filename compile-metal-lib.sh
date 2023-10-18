# https://stackoverflow.com/questions/70632495/how-to-build-apples-metal-cpp-example-using-cmake
# https://developer.apple.com/documentation/metal/shader_libraries/building_a_shader_library_by_precompiling_source_files?language=objc
xcrun -sdk macosx metal -o add.ir  -c add.metal
xcrun -sdk macosx metallib -o add.metallib  add.ir