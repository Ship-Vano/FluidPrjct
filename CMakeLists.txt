cmake_minimum_required(VERSION 3.20)
project(FluidGPU LANGUAGES CXX CUDA)

include_directories(${CMAKE_SOURCE_DIR}/InputData)

# Включаем расширенные лямбды для CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Set CUDA paths explicitly for Arch Linux's non-standard location
set(CMAKE_CUDA_ARCHITECTURES "native")

# Find CUDA Toolkit (включая cuDSS)
find_package(CUDAToolkit REQUIRED)

set(CUDSS_DIR "$ENV{HOME}/cudss")  # Путь к cuDSS

add_executable(FluidGPU
        src/FluidSim/utility.cu
        src/FluidSim/FluidSolver3D.cu
        src/FluidSim/test.cu
        src/json/jsoncpp.cpp
        main.cpp
)


# Упрощаем include-директории (CUDAToolkit сам найдёт пути)
target_include_directories(FluidGPU PRIVATE
        ${CMAKE_SOURCE_DIR}/src/json
        ${CMAKE_SOURCE_DIR}/src/FluidSim
        ${CMAKE_SOURCE_DIR}/InputData
        "${CUDSS_DIR}/include" # Заголовки cuDSS
)

target_link_directories(FluidGPU PRIVATE
        "${CUDSS_DIR}/lib"      # Библиотеки cuDSS
)

# Добавляем cuDSS в линковку
target_link_libraries(FluidGPU PRIVATE
        :libcudss.so.0
        CUDA::cusparse
        CUDA::cudart
)



# Убираем явные указания путей CUDA (find_package справится сам)
# target_link_directories можно не указывать, если нет проблем

# Set CUDA standard and properties
set_target_properties(FluidGPU PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)