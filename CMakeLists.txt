cmake_minimum_required(VERSION 3.18)
project(VCORE_RESONANCE LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Поиск CUDA
find_package(CUDA REQUIRED)

# Создание библиотеки ядра
add_library(vcore_engine SHARED 
    VCoreEngine.cpp 
    vcore_kernel.cu
)

set_target_properties(vcore_engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(vcore_engine PUBLIC ${CUDA_INCLUDE_DIRS})
