set(OpenCV_DIR /opt/homebrew/Cellar/opencv/4.8.1_1/include/opencv4)

# link OpenCV_INCLUDE_DIRS and OpenCV_LIBS
find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

# link metal
find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
find_library(APPLE_FWK_QUARTZ_CORE QuartzCore REQUIRED)
find_library(APPLE_FWK_METAL Metal REQUIRED)

add_library("RendererLib"
  "ARWebcam.cpp"
  "render/mesh.cpp"
  "render/TextureArray.cpp"
  "render/model.cpp"
  "render/MetalEngine.cpp"
)
add_library(stb "built-libs/stb/stb_image.cpp")
add_library(aapl "built-libs/aapl/AAPLMathUtilities.cpp")

add_executable(augmented_reality_on_webcam
  main.cpp)

add_custom_command(TARGET augmented_reality_on_webcam
  PRE_BUILD
  COMMAND /bin/sh ${PROJECT_SOURCE_DIR}/compile-metal-lib.sh
)

target_link_libraries(augmented_reality_on_webcam
  "RendererLib"
)

# link OpenCV libs
target_link_libraries(augmented_reality_on_webcam ${OpenCV_LIBS})

# Download and install assimp
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${PROJECT_SOURCE_DIR}/libs CACHE PATH "Missing description." FORCE)
FetchContent_Declare(assimp
GIT_REPOSITORY https://github.com/assimp/assimp.git
GIT_TAG master)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INJECT_DEBUG_POSTFIX OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(assimp)
target_include_directories("RendererLib" PRIVATE libs/assimp-src/include)
target_link_libraries("RendererLib" assimp)

# Link stb and metal for RendererLib
target_link_libraries("RendererLib" stb)
target_link_libraries("RendererLib" aapl)
target_include_directories("RendererLib"
  SYSTEM PUBLIC "built-libs/metal-cpp"
)
target_link_libraries("RendererLib"
  ${APPLE_FWK_FOUNDATION}
  ${APPLE_FWK_QUARTZ_CORE}
  ${APPLE_FWK_METAL}
)