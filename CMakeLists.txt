project(high-perf-cuda-op CXX CUDA)

cmake_minimum_required(VERSION 3.18)

find_package(CUDAToolkit REQUIRED)

# set Ampere Arch
set(CMAKE_CUDA_ARCHITECTURES 80)

include_directories(include)

add_subdirectory(sgemm)
