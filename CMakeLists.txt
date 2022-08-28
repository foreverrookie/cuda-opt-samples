project(cuda-opt-samples CXX CUDA)

cmake_minimum_required(VERSION 3.18)

find_package(CUDAToolkit REQUIRED)

# set Ampere Arch
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

include_directories(include)

add_subdirectory(sgemm)
