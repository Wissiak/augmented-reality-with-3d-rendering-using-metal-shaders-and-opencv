cd "$(dirname "$0")"
# https://stackoverflow.com/questions/70632495/how-to-build-apples-metal-cpp-example-using-cmake
# https://developer.apple.com/documentation/metal/shader_libraries/building_a_shader_library_by_precompiling_source_files?language=objc
xcrun -sdk macosx metal -o build/model.ir  -c src/model.metal
xcrun -sdk macosx metallib -o build/model.metallib  build/model.ir