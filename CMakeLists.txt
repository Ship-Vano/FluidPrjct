cmake_minimum_required(VERSION 3.20)
project(FluidGPU LANGUAGES CXX CUDA)

# Set CUDA paths explicitly for Arch Linux's non-standard location
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/opt/cuda/targets/x86_64-linux/include")

# Find CUDA (this will find cuSPARSE automatically)
find_package(CUDAToolkit REQUIRED)

add_executable(FluidGPU
        src/FluidSim/utility.cu
        src/FluidSim/FluidSolver2D.cu
        src/FluidSim/test.cu
        src/json/jsoncpp.cpp
        main.cpp
)

# Link CUDA libraries - KEY FIX for Arch Linux
target_link_libraries(FluidGPU PRIVATE
        CUDA::cusparse
        CUDA::cudart
)

# Explicit include paths for Arch Linux
target_include_directories(FluidGPU PRIVATE
        ${CMAKE_SOURCE_DIR}/src/json
        ${CMAKE_SOURCE_DIR}/src/FluidSim
        "/opt/cuda/targets/x86_64-linux/include"
)

# Additional linker paths for Arch Linux
target_link_directories(FluidGPU PRIVATE
        "/opt/cuda/targets/x86_64-linux/lib"
)

# Set CUDA standard and properties
set_target_properties(FluidGPU PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
)