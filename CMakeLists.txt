# CMakeLists.txt

# Older versions of CMake are likely to work just fine but, since
# I don't know where to cut off I just use the version I'm using
cmake_minimum_required(VERSION "3.17")

# name of this example project
project(augmented_reality_on_webcam)

# set OpenCV_DIR variable equal to the path to the cmake
# files within the previously installed opencv program
set(OpenCV_DIR /opt/homebrew/Cellar/opencv/4.8.1_1/include/opencv4)

# Tell compiler to use C++ 14 features which is needed because
# Clang version is often behind in the XCode installation
set(CMAKE_CXX_STANDARD 20)

# configure the necessary common CMake environment variables
# needed to include and link the OpenCV program into this
# demo project, namely OpenCV_INCLUDE_DIRS and OpenCV_LIBS
find_package( OpenCV REQUIRED )

# tell the build to include the headers from OpenCV
include_directories( ${OpenCV_INCLUDE_DIRS} )


find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
find_library(APPLE_FWK_QUARTZ_CORE QuartzCore REQUIRED)
find_library(APPLE_FWK_METAL Metal REQUIRED)

add_library("RendererLib"
    "mesh.cpp"
    "TextureArray.cpp"
    "model.cpp"
    "MetalEngine.cpp"
    "AAPLMathUtilities.cpp"
)
add_library(stb "stb/stb_image.cpp")
# specify the executable target to be built
add_executable(augmented_reality_on_webcam
  main.cpp)

#set_property(TARGET augmented_reality_on_webcam PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/build")

target_link_libraries(augmented_reality_on_webcam "RendererLib")

# tell it to link the executable target against OpenCV
target_link_libraries(augmented_reality_on_webcam ${OpenCV_LIBS})

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

target_link_libraries("RendererLib" stb)
target_include_directories("RendererLib"
  SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/metal-cpp
)
target_link_libraries("RendererLib"
  ${APPLE_FWK_FOUNDATION}
  ${APPLE_FWK_QUARTZ_CORE}
  ${APPLE_FWK_METAL}
)