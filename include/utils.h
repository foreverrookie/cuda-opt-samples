#ifndef CUDA_OPT_SAMPLES_UTILS_H
#define CUDA_OPT_SAMPLES_UTILS_H

#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

#define CUDA_CHECK_AND_FREE(device_ptr) \
    do {                                \
        if (device_ptr) {               \
            cudaFree(device_ptr);       \
        }                               \
    } while (0)

#define CUDA_CHECK(status)                                                           \
    do {                                                                             \
        auto ret = (status);                                                         \
        if (ret != 0) {                                                              \
            throw std::runtime_error("cuda failure: " + std::to_string(ret) + " (" + \
                                     cudaGetErrorString(ret) + ")" + " at " +        \
                                     __FILE__ + ":" + std::to_string(__LINE__));     \
        }                                                                            \
    } while (0)

// cuBLAS has not api like cudaGetErrorString()
#define CUBLAS_CHECK(status)                                                    \
    do {                                                                        \
        auto ret = (status);                                                    \
        if (ret != 0) {                                                         \
            throw std::runtime_error("cuBLAS failure: " + std::to_string(ret) + \
                                     " at " + __FILE__ + ":" +                  \
                                     std::to_string(__LINE__));                 \
        }                                                                       \
    } while (0)

// LDG.128, LDS.128
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2 *>(&(pointer))[0])

// i --- x-coordinate,      j --- y-coordinate
#define OFFSET(i, j, width) ((j) * (width) + i)

inline unsigned int DivUp(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

inline void RandomFloatVector(std::vector<float> &vec_f) {
    srand((unsigned int)(time(NULL)));
    std::for_each(vec_f.begin(), vec_f.end(),
                  [](float &f) { f = ((rand() % 1024) - 512.0f) / 256.0f; });
}

#endif