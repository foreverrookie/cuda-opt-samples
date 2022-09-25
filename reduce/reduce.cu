#include <iostream>
#include <utils.h>
#include <numeric>

#define PROFILE_REDUCE(expr, name, n)                                                                    \
    do {                                                                                                 \
        cudaEvent_t start, stop;                                                                         \
        cudaEventCreate(&start);                                                                         \
        cudaEventCreate(&stop);                                                                          \
        cudaEventRecord(start);                                                                          \
        expr;                                                                                            \
        cudaEventRecord(stop);                                                                           \
        cudaEventSynchronize(stop);                                                                      \
        float milliseconds = 0;                                                                          \
        cudaEventElapsedTime(&milliseconds, start, stop);                                                \
        double move_bytes = static_cast<double>(n * sizeof(float));                                      \
        std::cout << #name << " time: " << milliseconds << " ms, \t"                                     \
                  << "Bandwidth : " << move_bytes * 1e3 / milliseconds / (1024 * 1024 * 1024) << " GB/s" \
                  << std::endl;                                                                          \
        cudaEventDestroy(start);                                                                         \
        cudaEventDestroy(stop);                                                                          \
    } while (0)

__global__ void ReduceNaive(const float *src, float *dst, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    atomicAdd(dst, src[idx]);
}

__global__ void ReduceShm(const float *src, float *dst, int n) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;

    extern __shared__ float shm[];

    shm[tid] = 0.0f;

    if (idx >= n) {
        return;
    }

    shm[tid] = src[idx];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shm[tid] += shm[tid + stride];
        }
        __syncthreads();
    }

    // first thread in each block writes back to dst
    if (tid == 0) {
        atomicAdd(dst, shm[0]);
    }
}

__global__ void ReduceThreadLoop(const float *src, float *dst, int n, int thread_compute_nums) {
    int tid = threadIdx.x;
    int idx = blockDim.x * thread_compute_nums * blockIdx.x + tid;

    extern __shared__ float shm[];

    shm[tid] = 0.0f;
    if (idx < n) {
        shm[tid] = src[idx];
    }

    float tmp_sum = 0.0f;
    for (int i = 1; i < thread_compute_nums; i++) {
        int loop_idx = idx + i * blockDim.x;

        if (loop_idx < n) {
            tmp_sum += src[idx + i * blockDim.x];
        }
    }
    shm[tid] += tmp_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shm[tid] += shm[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(dst, shm[0]);
    }
}

template <int block_size>
__device__ inline void WarpReduce(volatile float *shm, int tid) {
    if (block_size >= 64) shm[tid] += shm[tid + 32];
    if (block_size >= 32) shm[tid] += shm[tid + 16];
    if (block_size >= 16) shm[tid] += shm[tid + 8];
    if (block_size >= 8) shm[tid] += shm[tid + 4];
    if (block_size >= 4) shm[tid] += shm[tid + 2];
    if (block_size >= 2) shm[tid] += shm[tid + 1];
}

template <int block_size>
__global__ void ReduceThreadLoopWarpUnroll(const float *src, float *dst, int n, int thread_compute_nums) {
    int tid = threadIdx.x;
    int idx = block_size * thread_compute_nums * blockIdx.x + tid;

    extern __shared__ float shm[];

    shm[tid] = 0.0f;
    if (idx < n) {
        shm[tid] = src[idx];
    }

    float tmp_sum = 0.0f;
    for (int i = 1; i < thread_compute_nums; i++) {
        int loop_idx = idx + i * block_size;

        if (loop_idx < n) {
            tmp_sum += src[idx + i * block_size];
        }
    }
    shm[tid] += tmp_sum;
    __syncthreads();

    if (block_size == 1024) {
        if (tid < 512) {
            shm[tid] += shm[tid + 512];
        }
        __syncthreads();
    }
    if (block_size >= 512) {
        if (tid < 256) {
            shm[tid] += shm[tid + 256];
        }
        __syncthreads();
    }
    if (block_size >= 256) {
        if (tid < 128) {
            shm[tid] += shm[tid + 128];
        }
        __syncthreads();
    }
    if (block_size >= 128) {
        if (tid < 64) {
            shm[tid] += shm[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        WarpReduce<block_size>(shm, tid);
    }
    __syncthreads();

    if (tid == 0) {
        atomicAdd(dst, shm[0]);
    }
}

int main(int argc, char *argv[]) {
    int n = 1024 * 1024 * 100;  // caution float error
    int thread_compute_nums = 8;

    if (argc > 3) {
        printf("usage: ./reduce [n] [thread_compute_nums]\n");
        exit(0);
    }
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (argc > 2) {
        thread_compute_nums = atoi(argv[2]);
    }

    float *in_dev = nullptr;
    float *out_dev = nullptr;

    CUDA_CHECK(cudaMalloc(&in_dev, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out_dev, sizeof(float)));

    std::vector<float> in_data(n);
    RandomFloatVector(in_data);

    float out = std::accumulate(in_data.begin(), in_data.end(), 0.0f, std::plus<float>());

    CUDA_CHECK(cudaMemcpy(in_dev, &in_data[0], n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(out_dev, 0, sizeof(float)));

    auto CheckOutput = [out](float *cuda_out_dev) -> void {
        float cuda_out;
        CUDA_CHECK(cudaMemcpy(&cuda_out, cuda_out_dev, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(cuda_out_dev, 0, sizeof(float)));

        if (std::fabs((cuda_out - out) / out) < 1e-3) {
            std::cout << "AC.\n\n";
        } else {
            std::cout << "NG. cuda reduce output: " << cuda_out
                      << ", cpp reduce output: " << out << std::endl;
            throw std::runtime_error("compute error.\n");
        }
    };

    // reduce_naive time: 171.083 ms,  Bandwidth : 2.28325 GB/s
    const int block_size = 256;
    dim3 grid(DivUp(n, block_size));
    PROFILE_REDUCE((ReduceNaive<<<grid, block_size>>>(in_dev, out_dev, n)), reduce_naive, n);
    CheckOutput(out_dev);

    // reduce_shm time: 1.55238 ms,    Bandwidth : 251.629 GB/s
    PROFILE_REDUCE((ReduceShm<<<grid, block_size, block_size * sizeof(float)>>>(in_dev, out_dev, n)),
                   reduce_shm, n);
    CheckOutput(out_dev);

    // reduce_thread_loop time : 0.586752 ms,   Bandwidth : 665.741 GB/s
    PROFILE_REDUCE((ReduceThreadLoop<<<DivUp(n, block_size * thread_compute_nums), block_size,
                                       block_size * sizeof(float)>>>(in_dev, out_dev, n, thread_compute_nums)),
                   reduce_thread_loop, n);
    CheckOutput(out_dev);

    // reduce_thread_loop_warp_unroll time: 0.585728 ms,       Bandwidth : 666.905 GB/s
    PROFILE_REDUCE((ReduceThreadLoopWarpUnroll<block_size><<<
                        DivUp(n, block_size * thread_compute_nums), block_size,
                        block_size * sizeof(float)>>>(in_dev, out_dev, n, thread_compute_nums)),
                   reduce_thread_loop_warp_unroll, n);
    CheckOutput(out_dev);

    CUDA_CHECK_AND_FREE(in_dev);
    CUDA_CHECK_AND_FREE(out_dev);
    return 0;
}