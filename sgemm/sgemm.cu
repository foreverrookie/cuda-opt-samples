#include <utils.h>

#include <iostream>

#include "cublas_v2.h"

void CheckCudaOutput(const std::vector<float> &cublas_output,
                     const std::vector<float> &cuda_output)
{
    int mn = cublas_output.size();
    for (int i = 0; i < mn; ++i)
    {
        // compare relative error
        if (std::fabs(cublas_output[i] - cuda_output[i]) / std::fabs(cublas_output[i]) > 1e-5)
        {
            std::cout << "index of error value: " << i
                      << ", cuBLAS value: " << cublas_output[i]
                      << ", my value: " << cuda_output[i] << std::endl;
            throw std::runtime_error("compute error!");
        }
    }
}

void CublasSgemm(float *mat_a, float *mat_b, float *mat_c,
                 std::vector<float> &cublas_c, int m, int k, int n)
{
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // warm up
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                             &alpha, mat_b, n, mat_a, k, &beta, mat_c, n));

    // NT, T - transpose, n - no transpose
    // C = A*B, A B C stored by column-major
    // if row-major output wanted, convert to CT = BT * AT
    PROFILE(
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                 &alpha, mat_b, n, mat_a, k, &beta, mat_c, n)),
        cublasSgemm, m, k, n);

    CUDA_CHECK(cudaMemcpy(cublas_c.data(), mat_c, m * n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

__global__ void SgemmNaive(float *mat_a, float *mat_b, float *mat_c, int m, int k, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx >= n || idy >= m)
    {
        return;
    }

    float cur = 0;

    for (int i = 0; i < k; ++i)
    {
        cur += mat_a[idy * k + i] * mat_b[i * n + idx];
    }

    mat_c[idy * n + idx] = cur;
}

__global__ void SgemmNaive2(float *mat_a, float *mat_b, float *mat_c, int m, int k, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx >= n || idy >= m)
        return;

    float cur = 0;
    float2 tmp_a;
    for (int i = 0; i < k; i += 2)
    {
        tmp_a = FETCH_FLOAT2(mat_a[idy * k + i]);
        float tmp_b1 = mat_b[i * n + idx];
        float tmp_b2 = mat_b[(i + 1) * n + idx];
        cur += tmp_a.x * tmp_b1 + tmp_a.y * tmp_b2;
    }

    mat_c[idy * n + idx] = cur;
}

__global__ void SgemmNaive4(float *mat_a, float *mat_b, float *mat_c, int m, int k, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx >= n || idy >= m)
        return;

    float cur = 0;
    float4 tmp_a;
    for (int i = 0; i < k; i += 4)
    {
        tmp_a = FETCH_FLOAT4(mat_a[idy * k + i]);
        float tmp_b1 = mat_b[i * n + idx];
        float tmp_b2 = mat_b[(i + 1) * n + idx];
        float tmp_b3 = mat_b[(i + 2) * n + idx];
        float tmp_b4 = mat_b[(i + 3) * n + idx];
        cur += tmp_a.x * tmp_b1 + tmp_a.y * tmp_b2 + tmp_a.z * tmp_b3 + tmp_a.w * tmp_b4;
    }

    mat_c[idy * n + idx] = cur;
}

__global__ void Sgemm16x16Divisible16(float *mat_a, float *mat_b, float *mat_c, int m, int k, int n)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int idx = blockIdx.x * blockDim.x + tx;
    int idy = blockIdx.y * blockDim.y + ty;

    __shared__ float mat_a_shm[16][16];
    __shared__ float mat_b_shm[16][16];

    float cur = 0.0f;

    for (int i = 0; i < k; i += 16)
    {
        // source index of global mat_a, wrong sample: idy * k + i * 16 + idx
        // source index of global mat_b, wrong sample: (i * 16 + ty) * n + idx
        mat_a_shm[ty][tx] = mat_a[idy * k + i + tx];
        mat_b_shm[ty][tx] = mat_b[(i + ty) * n + idx];

        __syncthreads();

        // inner product
        for (int j = 0; j < 16; ++j)
        {
            cur += mat_a_shm[ty][j] * mat_b_shm[j][tx];
        }
        // Synchronize to make sure all threads are done reading the submatrices
        // ! before overwriting them in the next iteration of the kb loop
        __syncthreads();
    }

    mat_c[idy * n + idx] = cur;
}

__global__ void Sgemm16x16(float *mat_a, float *mat_b, float *mat_c, int m, int k, int n)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int idx = blockIdx.x * blockDim.x + tx;
    int idy = blockIdx.y * blockDim.y + ty;

    if (idx >= n || idy >= m)
    {
        return;
    }

    float cur = 0;
    // boundary condition
    if (idx >= ((n >> 4) << 4) || idy >= ((m >> 4) << 4))
    {
        for (int i = 0; i < k; ++i)
        {
            cur += mat_a[idy * k + i] * mat_b[i * n + idx];
        }
        mat_c[idy * n + idx] = cur;
        return;
    }

    __shared__ float mat_a_shm[16][16];
    __shared__ float mat_b_shm[16][16];

    int k_up = (k >> 4) << 4;
    for (int i = 0; i < k_up; i += 16)
    {
        // source index of global mat_a, wrong sample: idy * k + i * 16 + idx
        // source index of global mat_b, wrong sample: (i * 16 + ty) * n + idx
        mat_a_shm[ty][tx] = mat_a[idy * k + i + tx];
        mat_b_shm[ty][tx] = mat_b[(i + ty) * n + idx];

        __syncthreads();

        // inner product
        for (int j = 0; j < 16; ++j)
        {
            cur += mat_a_shm[ty][j] * mat_b_shm[j][tx];
        }
        // Synchronize to make sure all threads are done reading the submatrices
        // ! before overwriting them in the next iteration of the kb loop
        __syncthreads();
    }

    mat_a_shm[ty][tx] = 0;
    mat_b_shm[ty][tx] = 0;

    // boundary condition
    if (tx < k - k_up)
    {
        mat_a_shm[ty][tx] = mat_a[idy * k + k_up + tx];
    }
    if (ty < k - k_up)
    {
        mat_b_shm[ty][tx] = mat_b[(k_up + ty) * n + idx];
    }
    // dont forget last sync
    __syncthreads();

    for (int j = 0; j < 16; ++j)
    {
        cur += mat_a_shm[ty][j] * mat_b_shm[j][tx];
    }

    mat_c[idy * n + idx] = cur;
}

// adopt default value in cutlass
// block tile: Mtile 128, Ntile 128, Ktile 8
// block size: 256, thread tile: 8 x 8
__global__ void Sgemm128x128Divisible128(float *mat_a, float *mat_b, float *mat_c, int m, int k, int n)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // start index of matrix C in current block
    int start_row_c = blockIdx.y * 128;
    int start_col_c = blockIdx.x * 128;

    __shared__ float __align__(16) mat_a_shm[8][128]; // 128 x 8
    __shared__ float __align__(16) mat_b_shm[8][128]; // 8 x 128

    float mat_a_frag[8]; // thread tile
    float mat_b_frag[8]; // thread tile
    float mat_c_accum[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            mat_c_accum[i][j] = 0.0f;
        }
    }

    int tid_mod_2 = tid % 2;
    int tid_div_2 = tid / 2;

    // main loop across Dim-k
    for (int i = 0; i < k; i += 8)
    {
        // load mat_a(LDG.128), gmem -> rf -> shm, transpose for consecutive access in outer product
        FETCH_FLOAT4(mat_a_frag[0]) =
            FETCH_FLOAT4(mat_a[OFFSET(i, start_row_c, k) +            /* main loop offset */
                               OFFSET(tid_mod_2 * 4, tid_div_2, k)]); /* thread offset */

        mat_a_shm[tid_mod_2 * 4][tid_div_2] = mat_a_frag[0];
        mat_a_shm[tid_mod_2 * 4 + 1][tid_div_2] = mat_a_frag[1];
        mat_a_shm[tid_mod_2 * 4 + 2][tid_div_2] = mat_a_frag[2];
        mat_a_shm[tid_mod_2 * 4 + 3][tid_div_2] = mat_a_frag[3];

        // load mat_b(LDG.128), gmem -> shm
        FETCH_FLOAT4(mat_b_shm[warp_id][lane_id * 4]) =
            FETCH_FLOAT4(mat_b[OFFSET(start_col_c, i, n) +        /* main loop offset */
                               OFFSET(lane_id * 4, warp_id, n)]); /* thread offset */

        __syncthreads();

#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            // 4 element load(LDS.128) in single instruction, two instructions
            FETCH_FLOAT4(mat_a_frag[0]) = FETCH_FLOAT4(mat_a_shm[j][(warp_id % 2) * 64 + (lane_id % 8) * 4]);
            FETCH_FLOAT4(mat_a_frag[4]) = FETCH_FLOAT4(mat_a_shm[j][(warp_id % 2) * 64 + (lane_id % 8) * 4 + 32]);

            FETCH_FLOAT4(mat_b_frag[0]) = FETCH_FLOAT4(mat_b_shm[j][(warp_id / 2) * 32 + (lane_id / 8) * 4]);
            FETCH_FLOAT4(mat_b_frag[4]) = FETCH_FLOAT4(mat_b_shm[j][(warp_id / 2) * 32 + (lane_id / 8) * 4 + 16]);

#pragma unroll
            for (int u = 0; u < 8; ++u)
            {
#pragma unroll
                for (int v = 0; v < 8; ++v)
                {
                    mat_c_accum[u][v] += mat_a_frag[u] * mat_b_frag[v];
                }
            }
        }
        __syncthreads();
    }

    int write_base_offset = OFFSET(start_col_c, start_row_c, n) +               /* block offset */
                            OFFSET((warp_id / 2) * 32, (warp_id % 2) * 64, n) + /* warp offset */
                            OFFSET((lane_id / 8) * 4, (lane_id % 8) * 4, n);    /* thread offset */
// write to mat_c
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // x o
        // o o
        FETCH_FLOAT4(mat_c[write_base_offset + i * n]) = FETCH_FLOAT4(mat_c_accum[i][0]);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // o x
        // o o
        FETCH_FLOAT4(mat_c[write_base_offset + 16 + i * n]) = FETCH_FLOAT4(mat_c_accum[i][4]);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // o o
        // x o
        FETCH_FLOAT4(mat_c[write_base_offset + 32 * n + i * n]) = FETCH_FLOAT4(mat_c_accum[i + 4][0]);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // o o
        // o x
        FETCH_FLOAT4(mat_c[write_base_offset + 32 * n + i * n + 16]) = FETCH_FLOAT4(mat_c_accum[i + 4][4]);
    }
}

__global__ void Sgemm128x128(float *mat_a, float *mat_b, float *mat_c, int m, int k, int n)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // start index of matrix C in current block
    int start_row_c = blockIdx.y * 128;
    int start_col_c = blockIdx.x * 128;

    __shared__ float __align__(16) mat_a_shm[8][128]; // 128 x 8
    __shared__ float __align__(16) mat_b_shm[8][128]; // 8 x 128

    FETCH_FLOAT4(mat_a_shm[warp_id][lane_id * 4]) = {0.0f, 0.0f, 0.0f, 0.0f};
    FETCH_FLOAT4(mat_b_shm[warp_id][lane_id * 4]) = {0.0f, 0.0f, 0.0f, 0.0f};

    __syncthreads();

    float mat_a_frag[8] = {0}; // thread tile
    float mat_b_frag[8] = {0}; // thread tile

    float mat_c_accum[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            mat_c_accum[i][j] = 0.0f;
        }
    }

    int tid_mod_2 = tid % 2;
    int tid_div_2 = tid / 2;

    // main loop across Dim-k
    for (int i = 0; i < k; i += 8)
    {
        // load mat_a(LDG.128), gmem -> shm, transpose for consecutive access in outer product
        int ver_offset = start_row_c + tid_div_2; // vertical offset
        int hor_offset = i + tid_mod_2 * 4;       // horizontal offset

        if (ver_offset < m)
        {
            for (int j = 0; j < 4; ++j)
            {
                if (j + hor_offset < k)
                {
                    mat_a_shm[tid_mod_2 * 4 + j][tid_div_2] = mat_a[OFFSET(j + hor_offset, ver_offset, k)];
                }
                else
                {
                    mat_a_shm[tid_mod_2 * 4 + j][tid_div_2] = 0.0f;
                }
            }
        }

        // load mat_b(LDG.128), gmem -> shm
        ver_offset = i + warp_id;
        hor_offset = start_col_c + lane_id * 4;

        if (ver_offset < k)
        {
            for (int j = 0; j < 4; ++j)
            {
                if (j + hor_offset < n)
                {
                    mat_b_shm[warp_id][lane_id * 4 + j] = mat_b[OFFSET(j + hor_offset, ver_offset, n)];
                }
                else
                {
                    mat_b_shm[warp_id][lane_id * 4 + j] = 0.0f;
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            // 4 element load(LDS.128) in single instruction, two instructions
            FETCH_FLOAT4(mat_a_frag[0]) = FETCH_FLOAT4(mat_a_shm[j][(warp_id % 2) * 64 + (lane_id % 8) * 4]);
            FETCH_FLOAT4(mat_a_frag[4]) = FETCH_FLOAT4(mat_a_shm[j][(warp_id % 2) * 64 + (lane_id % 8) * 4 + 32]);

            FETCH_FLOAT4(mat_b_frag[0]) = FETCH_FLOAT4(mat_b_shm[j][(warp_id / 2) * 32 + (lane_id / 8) * 4]);
            FETCH_FLOAT4(mat_b_frag[4]) = FETCH_FLOAT4(mat_b_shm[j][(warp_id / 2) * 32 + (lane_id / 8) * 4 + 16]);

#pragma unroll
            for (int u = 0; u < 8; ++u)
            {
#pragma unroll
                for (int v = 0; v < 8; ++v)
                {
                    mat_c_accum[u][v] += mat_a_frag[u] * mat_b_frag[v];
                }
            }
        }

        __syncthreads();
    }

    int ver_offset_c = start_row_c + (warp_id % 2) * 64 + (lane_id % 8) * 4;
    int hor_offset_c = start_col_c + (warp_id / 2) * 32 + (lane_id / 8) * 4;
// write to mat_c
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // x o
        // o o
        if (ver_offset_c + i < m)
        {
            for (int j = 0; (j < 4) && (j < n - hor_offset_c); ++j)
            {
                mat_c[OFFSET(hor_offset_c + j, ver_offset_c + i, n)] = mat_c_accum[i][j];
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // o x
        // o o
        if (ver_offset_c + i < m)
        {
            for (int j = 0; (j < 4) && (j < n - hor_offset_c - 16); ++j)
            {
                mat_c[OFFSET(hor_offset_c + 16 + j, ver_offset_c + i, n)] = mat_c_accum[i][4 + j];
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // o o
        // x o
        if (ver_offset_c + 32 + i < m)
        {
            for (int j = 0; (j < 4) && (j < n - hor_offset_c); ++j)
            {
                mat_c[OFFSET(hor_offset_c + j, ver_offset_c + 32 + i, n)] = mat_c_accum[i + 4][j];
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // o o
        // o x
        if (ver_offset_c + 32 + i < m)
        {
            for (int j = 0; (j < 4) && (j < n - hor_offset_c - 16); ++j)
            {
                mat_c[OFFSET(hor_offset_c + j + 16, ver_offset_c + 32 + i, n)] = mat_c_accum[i + 4][4 + j];
            }
        }
    }
}

__global__ void Sgemm128x128Buf2Divisible128(float *mat_a, float *mat_b, float *mat_c, int m, int k, int n)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // start index of matrix C in current block
    int start_row_c = blockIdx.y * 128;
    int start_col_c = blockIdx.x * 128;

    __shared__ float __align__(16) mat_a_shm[2][8][128]; // 128 x 8, double buffer
    __shared__ float __align__(16) mat_b_shm[2][8][128]; // 8 x 128, double buffer

    float mat_a_ldg128[4]; // load global(LDG.128), double buffer
    float mat_b_ldg128[4]; // load global(LDG.128), double buffer

    float mat_a_frag[2][8]; // thread tile, double buffer
    float mat_b_frag[2][8]; // thread tile, double buffer
    float mat_c_accum[8][8] = {0};

    int tid_mod_2 = tid % 2;
    int tid_div_2 = tid / 2;

    // produce: load first smem buffer(step 1 of 2), gmem -> smem
    {
        // load mat_a(LDG.128), gmem -> rf -> shm, transpose for consecutive access in outer product
        FETCH_FLOAT4(mat_a_ldg128[0]) =
            FETCH_FLOAT4(mat_a[OFFSET(0, start_row_c, k) +            /* main loop offset */
                               OFFSET(tid_mod_2 * 4, tid_div_2, k)]); /* thread offset */

        mat_a_shm[0][tid_mod_2 * 4][tid_div_2] = mat_a_ldg128[0];
        mat_a_shm[0][tid_mod_2 * 4 + 1][tid_div_2] = mat_a_ldg128[1];
        mat_a_shm[0][tid_mod_2 * 4 + 2][tid_div_2] = mat_a_ldg128[2];
        mat_a_shm[0][tid_mod_2 * 4 + 3][tid_div_2] = mat_a_ldg128[3];

        // load mat_b(LDG.128), gmem -> shm
        FETCH_FLOAT4(mat_b_shm[0][warp_id][lane_id * 4]) =
            FETCH_FLOAT4(mat_b[OFFSET(start_col_c, 0, n) +        /* main loop offset */
                               OFFSET(lane_id * 4, warp_id, n)]); /* thread offset */
    }
    __syncthreads();

    // produce: load first rf buffer(step 1 of 8), smem -> rf
    {
        FETCH_FLOAT4(mat_a_frag[0][0]) =
            FETCH_FLOAT4(mat_a_shm[0][0][(warp_id % 2) * 64 + (lane_id % 8) * 4]);
        FETCH_FLOAT4(mat_a_frag[0][4]) =
            FETCH_FLOAT4(mat_a_shm[0][0][(warp_id % 2) * 64 + (lane_id % 8) * 4 + 32]);

        FETCH_FLOAT4(mat_b_frag[0][0]) =
            FETCH_FLOAT4(mat_b_shm[0][0][(warp_id / 2) * 32 + (lane_id / 8) * 4]);
        FETCH_FLOAT4(mat_b_frag[0][4]) =
            FETCH_FLOAT4(mat_b_shm[0][0][(warp_id / 2) * 32 + (lane_id / 8) * 4 + 16]);
    }

    int smem_consume_id = 0;

    // i start from 8, beacuse first tile is loaded from gmem to smem
    for (int i = 8; i < k; i += 8)
    {
        // smem produce: load next mat_a & mat_b tile, gmem -> rf
        {
            FETCH_FLOAT4(mat_a_ldg128[0]) =
                FETCH_FLOAT4(mat_a[OFFSET(i, start_row_c, k) +            /* main loop offset */
                                   OFFSET(tid_mod_2 * 4, tid_div_2, k)]); /* thread offset */

            FETCH_FLOAT4(mat_b_ldg128[0]) =
                FETCH_FLOAT4(mat_b[OFFSET(start_col_c, i, n) +        /* main loop offset */
                                   OFFSET(lane_id * 4, warp_id, n)]); /* thread offset */
        }

#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            // smem produce: store next mat_a & mat_b tile, rf -> smem
            if (j == 7)
            {
                mat_a_shm[smem_consume_id ^ 1][tid_mod_2 * 4][tid_div_2] = mat_a_ldg128[0];
                mat_a_shm[smem_consume_id ^ 1][tid_mod_2 * 4 + 1][tid_div_2] = mat_a_ldg128[1];
                mat_a_shm[smem_consume_id ^ 1][tid_mod_2 * 4 + 2][tid_div_2] = mat_a_ldg128[2];
                mat_a_shm[smem_consume_id ^ 1][tid_mod_2 * 4 + 3][tid_div_2] = mat_a_ldg128[3];

                FETCH_FLOAT4(mat_b_shm[smem_consume_id ^ 1][warp_id][lane_id * 4]) =
                    FETCH_FLOAT4(mat_b_ldg128[0]);

                __syncthreads();

                smem_consume_id ^= 1;
            }

            // rf produce: load next mat_a & mat_b fragment, smem -> rf
            // 4 element load(LDS.128) in single instruction, two instructions
            FETCH_FLOAT4(mat_a_frag[(j + 1) % 2][0]) =
                FETCH_FLOAT4(mat_a_shm[smem_consume_id][(j + 1) % 8][(warp_id % 2) * 64 + (lane_id % 8) * 4]);
            FETCH_FLOAT4(mat_a_frag[(j + 1) % 2][4]) =
                FETCH_FLOAT4(mat_a_shm[smem_consume_id][(j + 1) % 8][(warp_id % 2) * 64 + (lane_id % 8) * 4 + 32]);

            FETCH_FLOAT4(mat_b_frag[(j + 1) % 2][0]) =
                FETCH_FLOAT4(mat_b_shm[smem_consume_id][(j + 1) % 8][(warp_id / 2) * 32 + (lane_id / 8) * 4]);
            FETCH_FLOAT4(mat_b_frag[(j + 1) % 2][4]) =
                FETCH_FLOAT4(mat_b_shm[smem_consume_id][(j + 1) % 8][(warp_id / 2) * 32 + (lane_id / 8) * 4 + 16]);

// rf consume
#pragma unroll
            for (int u = 0; u < 8; ++u)
            {
#pragma unroll
                for (int v = 0; v < 8; ++v)
                {
                    mat_c_accum[u][v] += mat_a_frag[j % 2][u] * mat_b_frag[j % 2][v];
                }
            }
        }
    }

// consume last fragment
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        // produce: load next mat_a & mat_b fragment, smem -> rf
        if (j < 7)
        {
            FETCH_FLOAT4(mat_a_frag[(j + 1) % 2][0]) =
                FETCH_FLOAT4(mat_a_shm[smem_consume_id][(j + 1) % 8][(warp_id % 2) * 64 + (lane_id % 8) * 4]);
            FETCH_FLOAT4(mat_a_frag[(j + 1) % 2][4]) =
                FETCH_FLOAT4(mat_a_shm[smem_consume_id][(j + 1) % 8][(warp_id % 2) * 64 + (lane_id % 8) * 4 + 32]);

            FETCH_FLOAT4(mat_b_frag[(j + 1) % 2][0]) =
                FETCH_FLOAT4(mat_b_shm[smem_consume_id][(j + 1) % 8][(warp_id / 2) * 32 + (lane_id / 8) * 4]);
            FETCH_FLOAT4(mat_b_frag[(j + 1) % 2][4]) =
                FETCH_FLOAT4(mat_b_shm[smem_consume_id][(j + 1) % 8][(warp_id / 2) * 32 + (lane_id / 8) * 4 + 16]);
        }

#pragma unroll
        for (int u = 0; u < 8; ++u)
        {
#pragma unroll
            for (int v = 0; v < 8; ++v)
            {
                mat_c_accum[u][v] += mat_a_frag[j % 2][u] * mat_b_frag[j % 2][v];
            }
        }
    }

    int write_base_offset = OFFSET(start_col_c, start_row_c, n) +               /* block offset */
                            OFFSET((warp_id / 2) * 32, (warp_id % 2) * 64, n) + /* warp offset */
                            OFFSET((lane_id / 8) * 4, (lane_id % 8) * 4, n);    /* thread offset */
// write to mat_c
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // x o
        // o o
        FETCH_FLOAT4(mat_c[write_base_offset + i * n]) = FETCH_FLOAT4(mat_c_accum[i][0]);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // o x
        // o o
        FETCH_FLOAT4(mat_c[write_base_offset + 16 + i * n]) = FETCH_FLOAT4(mat_c_accum[i][4]);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // o o
        // x o
        FETCH_FLOAT4(mat_c[write_base_offset + 32 * n + i * n]) = FETCH_FLOAT4(mat_c_accum[i + 4][0]);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // o o
        // o x
        FETCH_FLOAT4(mat_c[write_base_offset + 32 * n + i * n + 16]) = FETCH_FLOAT4(mat_c_accum[i + 4][4]);
    }
}

__global__ void Sgemm128x128Buf2(float *mat_a, float *mat_b, float *mat_c, int m, int k, int n)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // start index of matrix C in current block
    int start_row_c = blockIdx.y * 128;
    int start_col_c = blockIdx.x * 128;

    __shared__ float __align__(16) mat_a_shm[2][8][128]; // 128 x 8, double buffer
    __shared__ float __align__(16) mat_b_shm[2][8][128]; // 8 x 128, double buffer

    float mat_a_ldg128[4]; // load global(LDG.128), double buffer
    float mat_b_ldg128[4]; // load global(LDG.128), double buffer

    float mat_a_frag[2][8]; // thread tile, double buffer
    float mat_b_frag[2][8]; // thread tile, double buffer
    float mat_c_accum[8][8] = {0};

    int tid_mod_2 = tid % 2;
    int tid_div_2 = tid / 2;

    int ver_offset = start_row_c + tid_div_2; // vertical offset
    int hor_offset = tid_mod_2 * 4;           // horizontal offset
    // produce: load first smem buffer(step 1 of 2), gmem -> smem
    {
        // load mat_a(LDG.128), gmem -> rf -> shm, transpose for consecutive access in outer product
        if (ver_offset < m)
        {
            for (int j = 0; j < 4; ++j)
            {
                if (j + hor_offset < k)
                {
                    mat_a_shm[0][tid_mod_2 * 4 + j][tid_div_2] = mat_a[OFFSET(j + hor_offset, ver_offset, k)];
                }
                else
                {
                    mat_a_shm[0][tid_mod_2 * 4 + j][tid_div_2] = 0.0f;
                }
            }
        }

        ver_offset = warp_id;
        hor_offset = start_col_c + lane_id * 4;

        // load mat_b(LDG.128), gmem -> shm
        if (ver_offset < k)
        {
            for (int j = 0; j < 4; ++j)
            {
                if (j + hor_offset < n)
                {
                    mat_b_shm[0][warp_id][lane_id * 4 + j] = mat_b[OFFSET(j + hor_offset, ver_offset, n)];
                }
                else
                {
                    mat_b_shm[0][warp_id][lane_id * 4 + j] = 0.0f;
                }
            }
        }
    }
    __syncthreads();

    // produce: load first rf buffer(step 1 of 8), smem -> rf
    {
        FETCH_FLOAT4(mat_a_frag[0][0]) =
            FETCH_FLOAT4(mat_a_shm[0][0][(warp_id % 2) * 64 + (lane_id % 8) * 4]);
        FETCH_FLOAT4(mat_a_frag[0][4]) =
            FETCH_FLOAT4(mat_a_shm[0][0][(warp_id % 2) * 64 + (lane_id % 8) * 4 + 32]);

        FETCH_FLOAT4(mat_b_frag[0][0]) =
            FETCH_FLOAT4(mat_b_shm[0][0][(warp_id / 2) * 32 + (lane_id / 8) * 4]);
        FETCH_FLOAT4(mat_b_frag[0][4]) =
            FETCH_FLOAT4(mat_b_shm[0][0][(warp_id / 2) * 32 + (lane_id / 8) * 4 + 16]);
    }

    int smem_consume_id = 0;

    // i start from 8, beacuse first tile is loaded from gmem to smem
    for (int i = 8; i < k; i += 8)
    {
        // smem produce: load next mat_a & mat_b tile, gmem -> rf
        {
            ver_offset = start_row_c + tid_div_2; // vertical offset
            hor_offset = i + tid_mod_2 * 4;       // horizontal offset

            if (ver_offset < m)
            {
                for (int j = 0; j < 4; ++j)
                {
                    if (j + hor_offset < k)
                    {
                        mat_a_ldg128[j] = mat_a[OFFSET(j + hor_offset, ver_offset, k)];
                    }
                    else
                    {
                        mat_a_ldg128[j] = 0.0f;
                    }
                }
            }

            ver_offset = i + warp_id;
            hor_offset = start_col_c + lane_id * 4;

            if (ver_offset < k)
            {
                for (int j = 0; j < 4; ++j)
                {
                    if (j + hor_offset < n)
                    {
                        mat_b_ldg128[j] = mat_b[OFFSET(j + hor_offset, ver_offset, n)];
                    }
                    else
                    {
                        mat_b_ldg128[j] = 0.0f;
                    }
                }
            }
        }

#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            // smem produce: store next mat_a & mat_b tile, rf -> smem
            if (j == 7)
            {
                mat_a_shm[smem_consume_id ^ 1][tid_mod_2 * 4][tid_div_2] = mat_a_ldg128[0];
                mat_a_shm[smem_consume_id ^ 1][tid_mod_2 * 4 + 1][tid_div_2] = mat_a_ldg128[1];
                mat_a_shm[smem_consume_id ^ 1][tid_mod_2 * 4 + 2][tid_div_2] = mat_a_ldg128[2];
                mat_a_shm[smem_consume_id ^ 1][tid_mod_2 * 4 + 3][tid_div_2] = mat_a_ldg128[3];

                FETCH_FLOAT4(mat_b_shm[smem_consume_id ^ 1][warp_id][lane_id * 4]) =
                    FETCH_FLOAT4(mat_b_ldg128[0]);

                __syncthreads();

                smem_consume_id ^= 1;
            }

            // rf produce: load next mat_a & mat_b fragment, smem -> rf
            // 4 element load(LDS.128) in single instruction, two instructions
            FETCH_FLOAT4(mat_a_frag[(j + 1) % 2][0]) =
                FETCH_FLOAT4(mat_a_shm[smem_consume_id][(j + 1) % 8][(warp_id % 2) * 64 + (lane_id % 8) * 4]);
            FETCH_FLOAT4(mat_a_frag[(j + 1) % 2][4]) =
                FETCH_FLOAT4(mat_a_shm[smem_consume_id][(j + 1) % 8][(warp_id % 2) * 64 + (lane_id % 8) * 4 + 32]);

            FETCH_FLOAT4(mat_b_frag[(j + 1) % 2][0]) =
                FETCH_FLOAT4(mat_b_shm[smem_consume_id][(j + 1) % 8][(warp_id / 2) * 32 + (lane_id / 8) * 4]);
            FETCH_FLOAT4(mat_b_frag[(j + 1) % 2][4]) =
                FETCH_FLOAT4(mat_b_shm[smem_consume_id][(j + 1) % 8][(warp_id / 2) * 32 + (lane_id / 8) * 4 + 16]);

// rf consume
#pragma unroll
            for (int u = 0; u < 8; ++u)
            {
#pragma unroll
                for (int v = 0; v < 8; ++v)
                {
                    mat_c_accum[u][v] += mat_a_frag[j % 2][u] * mat_b_frag[j % 2][v];
                }
            }
        }
    }

// consume last fragment
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        // produce: load next mat_a & mat_b fragment, smem -> rf
        if (j < 7)
        {
            FETCH_FLOAT4(mat_a_frag[(j + 1) % 2][0]) =
                FETCH_FLOAT4(mat_a_shm[smem_consume_id][(j + 1) % 8][(warp_id % 2) * 64 + (lane_id % 8) * 4]);
            FETCH_FLOAT4(mat_a_frag[(j + 1) % 2][4]) =
                FETCH_FLOAT4(mat_a_shm[smem_consume_id][(j + 1) % 8][(warp_id % 2) * 64 + (lane_id % 8) * 4 + 32]);

            FETCH_FLOAT4(mat_b_frag[(j + 1) % 2][0]) =
                FETCH_FLOAT4(mat_b_shm[smem_consume_id][(j + 1) % 8][(warp_id / 2) * 32 + (lane_id / 8) * 4]);
            FETCH_FLOAT4(mat_b_frag[(j + 1) % 2][4]) =
                FETCH_FLOAT4(mat_b_shm[smem_consume_id][(j + 1) % 8][(warp_id / 2) * 32 + (lane_id / 8) * 4 + 16]);
        }

#pragma unroll
        for (int u = 0; u < 8; ++u)
        {
#pragma unroll
            for (int v = 0; v < 8; ++v)
            {
                mat_c_accum[u][v] += mat_a_frag[j % 2][u] * mat_b_frag[j % 2][v];
            }
        }
    }

    int ver_offset_c = start_row_c + (warp_id % 2) * 64 + (lane_id % 8) * 4;
    int hor_offset_c = start_col_c + (warp_id / 2) * 32 + (lane_id / 8) * 4;
// write to mat_c
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // x o
        // o o

        if (ver_offset_c + i < m)
        {
            for (int j = 0; (j < 4) && (j < n - hor_offset_c); ++j)
            {
                mat_c[OFFSET(hor_offset_c + j, ver_offset_c + i, n)] = mat_c_accum[i][j];
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // o x
        // o o

        if (ver_offset_c + i < m)
        {
            for (int j = 0; (j < 4) && (j < n - hor_offset_c - 16); ++j)
            {
                mat_c[OFFSET(hor_offset_c + 16 + j, ver_offset_c + i, n)] = mat_c_accum[i][4 + j];
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // o o
        // x o

        if (ver_offset_c + 32 + i < m)
        {
            for (int j = 0; (j < 4) && (j < n - hor_offset_c); ++j)
            {
                mat_c[OFFSET(hor_offset_c + j, ver_offset_c + 32 + i, n)] = mat_c_accum[i + 4][j];
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // o o
        // o x

        if (ver_offset_c + 32 + i < m)
        {
            for (int j = 0; (j < 4) && (j < n - hor_offset_c - 16); ++j)
            {
                mat_c[OFFSET(hor_offset_c + j + 16, ver_offset_c + 32 + i, n)] = mat_c_accum[i + 4][4 + j];
            }
        }
    }
}

// opt based on Sgemm128x128Buf2
// inner warp shape: 4x8, thread map: "row" major
// inter warp shape: 4x2, warp map: "row" major
__global__ void Sgemm128x128Buf2OptDivisible128(float *mat_a, float *mat_b, float *mat_c, int m, int k, int n)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // start index of matrix C in current block
    int start_row_c = blockIdx.y * 128;
    int start_col_c = blockIdx.x * 128;

    __shared__ float __align__(16) mat_a_shm[2][8][128]; // 128 x 8, double buffer
    __shared__ float __align__(16) mat_b_shm[2][8][128]; // 8 x 128, double buffer

    float mat_a_ldg128[4]; // load global(LDG.128), double buffer
    float mat_b_ldg128[4]; // load global(LDG.128), double buffer

    float mat_a_frag[2][8]; // thread tile, double buffer
    float mat_b_frag[2][8]; // thread tile, double buffer
    float mat_c_accum[8][8] = {0};

    int tid_mod_2 = tid % 2;
    int tid_div_2 = tid / 2;

    // produce: load first smem buffer(step 1 of 2), gmem -> smem
    {
        // load mat_a(LDG.128), gmem -> rf -> shm, transpose for consecutive access in outer product
        FETCH_FLOAT4(mat_a_ldg128[0]) =
            FETCH_FLOAT4(mat_a[OFFSET(0, start_row_c, k) +            /* main loop offset */
                               OFFSET(tid_mod_2 * 4, tid_div_2, k)]); /* thread offset */

        mat_a_shm[0][tid_mod_2 * 4][tid_div_2] = mat_a_ldg128[0];
        mat_a_shm[0][tid_mod_2 * 4 + 1][tid_div_2] = mat_a_ldg128[1];
        mat_a_shm[0][tid_mod_2 * 4 + 2][tid_div_2] = mat_a_ldg128[2];
        mat_a_shm[0][tid_mod_2 * 4 + 3][tid_div_2] = mat_a_ldg128[3];

        // load mat_b(LDG.128), gmem -> shm
        FETCH_FLOAT4(mat_b_shm[0][warp_id][lane_id * 4]) =
            FETCH_FLOAT4(mat_b[OFFSET(start_col_c, 0, n) +        /* main loop offset */
                               OFFSET(lane_id * 4, warp_id, n)]); /* thread offset */
    }
    __syncthreads();

    int shm_a_offset = (warp_id / 2) * 32 + (lane_id / 8) * 4;
    int shm_b_offset = (warp_id % 2) * 64 + (lane_id % 8) * 4;
    // produce: load first rf buffer(step 1 of 8), smem -> rf
    {
        FETCH_FLOAT4(mat_a_frag[0][0]) =
            FETCH_FLOAT4(mat_a_shm[0][0][shm_a_offset]);
        FETCH_FLOAT4(mat_a_frag[0][4]) =
            FETCH_FLOAT4(mat_a_shm[0][0][shm_a_offset + 16]);

        FETCH_FLOAT4(mat_b_frag[0][0]) =
            FETCH_FLOAT4(mat_b_shm[0][0][shm_b_offset]);
        FETCH_FLOAT4(mat_b_frag[0][4]) =
            FETCH_FLOAT4(mat_b_shm[0][0][shm_b_offset + 32]);
    }

    int smem_consume_id = 0;

    // i start from 8, beacuse first tile is loaded from gmem to smem
    for (int i = 8; i < k; i += 8)
    {
        // smem produce: load next mat_a & mat_b tile, gmem -> rf
        {
            FETCH_FLOAT4(mat_a_ldg128[0]) =
                FETCH_FLOAT4(mat_a[OFFSET(i, start_row_c, k) +            /* main loop offset */
                                   OFFSET(tid_mod_2 * 4, tid_div_2, k)]); /* thread offset */

            FETCH_FLOAT4(mat_b_ldg128[0]) =
                FETCH_FLOAT4(mat_b[OFFSET(start_col_c, i, n) +        /* main loop offset */
                                   OFFSET(lane_id * 4, warp_id, n)]); /* thread offset */
        }

#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            // smem produce: store next mat_a & mat_b tile, rf -> smem
            if (j == 7)
            {
                mat_a_shm[smem_consume_id ^ 1][tid_mod_2 * 4][tid_div_2] = mat_a_ldg128[0];
                mat_a_shm[smem_consume_id ^ 1][tid_mod_2 * 4 + 1][tid_div_2] = mat_a_ldg128[1];
                mat_a_shm[smem_consume_id ^ 1][tid_mod_2 * 4 + 2][tid_div_2] = mat_a_ldg128[2];
                mat_a_shm[smem_consume_id ^ 1][tid_mod_2 * 4 + 3][tid_div_2] = mat_a_ldg128[3];

                FETCH_FLOAT4(mat_b_shm[smem_consume_id ^ 1][warp_id][lane_id * 4]) =
                    FETCH_FLOAT4(mat_b_ldg128[0]);

                __syncthreads();

                smem_consume_id ^= 1;
            }

            // rf produce: load next mat_a & mat_b fragment, smem -> rf
            // 4 element load(LDS.128) in single instruction, two instructions
            FETCH_FLOAT4(mat_a_frag[(j + 1) % 2][0]) =
                FETCH_FLOAT4(mat_a_shm[smem_consume_id][(j + 1) % 8][shm_a_offset]);
            FETCH_FLOAT4(mat_a_frag[(j + 1) % 2][4]) =
                FETCH_FLOAT4(mat_a_shm[smem_consume_id][(j + 1) % 8][shm_a_offset + 16]);

            FETCH_FLOAT4(mat_b_frag[(j + 1) % 2][0]) =
                FETCH_FLOAT4(mat_b_shm[smem_consume_id][(j + 1) % 8][shm_b_offset]);
            FETCH_FLOAT4(mat_b_frag[(j + 1) % 2][4]) =
                FETCH_FLOAT4(mat_b_shm[smem_consume_id][(j + 1) % 8][shm_b_offset + 32]);

// rf consume
#pragma unroll
            for (int u = 0; u < 8; ++u)
            {
#pragma unroll
                for (int v = 0; v < 8; ++v)
                {
                    mat_c_accum[u][v] += mat_a_frag[j % 2][u] * mat_b_frag[j % 2][v];
                }
            }
        }
    }

// consume last fragment
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        // produce: load next mat_a & mat_b fragment, smem -> rf
        if (j < 7)
        {
            FETCH_FLOAT4(mat_a_frag[(j + 1) % 2][0]) =
                FETCH_FLOAT4(mat_a_shm[smem_consume_id][(j + 1) % 8][shm_a_offset]);
            FETCH_FLOAT4(mat_a_frag[(j + 1) % 2][4]) =
                FETCH_FLOAT4(mat_a_shm[smem_consume_id][(j + 1) % 8][shm_a_offset + 16]);

            FETCH_FLOAT4(mat_b_frag[(j + 1) % 2][0]) =
                FETCH_FLOAT4(mat_b_shm[smem_consume_id][(j + 1) % 8][shm_b_offset]);
            FETCH_FLOAT4(mat_b_frag[(j + 1) % 2][4]) =
                FETCH_FLOAT4(mat_b_shm[smem_consume_id][(j + 1) % 8][shm_b_offset + 32]);
        }

#pragma unroll
        for (int u = 0; u < 8; ++u)
        {
#pragma unroll
            for (int v = 0; v < 8; ++v)
            {
                mat_c_accum[u][v] += mat_a_frag[j % 2][u] * mat_b_frag[j % 2][v];
            }
        }
    }

    int write_base_offset = OFFSET(start_col_c, start_row_c, n) +               /* block offset */
                            OFFSET((warp_id % 2) * 64, (warp_id / 2) * 32, n) + /* warp offset */
                            OFFSET((lane_id % 8) * 4, (lane_id / 8) * 4, n);    /* thread offset */
// write to mat_c
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // x o
        // o o
        FETCH_FLOAT4(mat_c[write_base_offset + i * n]) = FETCH_FLOAT4(mat_c_accum[i][0]);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // o x
        // o o
        FETCH_FLOAT4(mat_c[write_base_offset + 32 + i * n]) = FETCH_FLOAT4(mat_c_accum[i][4]);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // o o
        // x o
        FETCH_FLOAT4(mat_c[write_base_offset + 16 * n + i * n]) = FETCH_FLOAT4(mat_c_accum[i + 4][0]);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // o o
        // o x
        FETCH_FLOAT4(mat_c[write_base_offset + 16 * n + i * n + 32]) = FETCH_FLOAT4(mat_c_accum[i + 4][4]);
    }
}

int main(int argc, char *argv[])
{
    int m = 4096;
    int k = 4096;
    int n = 4096;
    if (argc != 4 && argc != 1)
    {
        printf("usage: ./sgemm [m] [k] [n]\n");
        exit(0);
    }
    if (argc == 4)
    {
        m = atoi(argv[1]);
        k = atoi(argv[2]);
        n = atoi(argv[3]);
    }

    float *mat_a_dev = nullptr;
    float *mat_b_dev = nullptr;
    float *mat_c_dev = nullptr;

    CUDA_CHECK(cudaMalloc(&mat_a_dev, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mat_b_dev, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mat_c_dev, m * n * sizeof(float)));

    std::vector<float> mat_a(m * k);
    std::vector<float> mat_b(k * n);
    RandomFloatVector(mat_a);
    RandomFloatVector(mat_b);

    std::vector<float> cublas_c(m * n);
    std::vector<float> cuda_c(m * n);

    CUDA_CHECK(cudaMemcpy(mat_a_dev, &mat_a[0], m * k * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mat_b_dev, &mat_b[0], k * n * sizeof(float),
                          cudaMemcpyHostToDevice));

    {
        // pole position, cuBLAS
        // cublasSgemm time: 7.18746 ms,   FP32 Perf: 19.1221 TFlops
        CublasSgemm(mat_a_dev, mat_b_dev, mat_c_dev, cublas_c, m, k, n);
        std::cout << std::endl;
    }

    dim3 block1(16, 16);
    dim3 grid1(DivUp(n, 16), DivUp(m, 16));
    {
        // sgemm_naive time: 72.7644 ms,    FP32 Perf: 1.88882 TFlops
        PROFILE((SgemmNaive<<<grid1, block1>>>(mat_a_dev, mat_b_dev, mat_c_dev, m, k, n)),
                sgemm_naive, m, k, n);

        CUDA_CHECK(cudaMemcpy(cuda_c.data(), mat_c_dev, m * n * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());

        CheckCudaOutput(cublas_c, cuda_c);
        std::cout << "Check SgemmNaive Output: Pass! \n";
        std::cout << std::endl;
    }
    if (k % 2 == 0)
    {
        // sgemm_naive2 time: 60.8432 ms,  FP32 Perf: 2.25891 TFlops
        PROFILE((SgemmNaive2<<<grid1, block1>>>(mat_a_dev, mat_b_dev, mat_c_dev, m, k, n)),
                sgemm_naive2, m, k, n);

        CUDA_CHECK(cudaMemcpy(cuda_c.data(), mat_c_dev, m * n * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());

        CheckCudaOutput(cublas_c, cuda_c);
        std::cout << "Check SgemmNaive2 Output: Pass! \n";
        std::cout << std::endl;
    }
    if (k % 4 == 0)
    {
        // sgemm_naive4 time: 54.8171 ms,  FP32 Perf: 2.50723 TFlops
        PROFILE((SgemmNaive4<<<grid1, block1>>>(mat_a_dev, mat_b_dev, mat_c_dev, m, k, n)),
                sgemm_naive4, m, k, n);

        CUDA_CHECK(cudaMemcpy(cuda_c.data(), mat_c_dev, m * n * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());

        CheckCudaOutput(cublas_c, cuda_c);
        std::cout << "Check SgemmNaive4 Output: Pass! \n";
        std::cout << std::endl;
    }

    {
        // sgemm_16x16_Divisible16 time: 49.1357 ms,   FP32 Perf: 2.79713 TFlops
        // sgemm_16x16             time: 50.4187 ms,    FP32 Perf: 2.72329 TFlops
        PROFILE((Sgemm16x16<<<grid1, block1>>>(mat_a_dev, mat_b_dev, mat_c_dev, m, k, n)),
                sgemm_16x16, m, k, n);

        CUDA_CHECK(cudaMemcpy(cuda_c.data(), mat_c_dev, m * n * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());

        CheckCudaOutput(cublas_c, cuda_c);
        std::cout << "Check Sgemm16x16 Output: Pass! \n";
        std::cout << std::endl;
    }

    dim3 block2(256, 1);
    dim3 grid2(DivUp(n, 128), DivUp(m, 128));
    {
        // sgemm_128x128_Divisible128 time: 7.55507 ms,   FP32 Perf: 18.1916 TFlops
        // sgemm_128x128              time: 9.25184 ms,   FP32 Perf: 14.8553 TFlops
        PROFILE((Sgemm128x128<<<grid2, block2>>>(mat_a_dev, mat_b_dev, mat_c_dev, m, k, n)),
                sgemm_128x128, m, k, n);

        CUDA_CHECK(cudaMemcpy(cuda_c.data(), mat_c_dev, m * n * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());

        CheckCudaOutput(cublas_c, cuda_c);
        std::cout << "Check Sgemm128x128 Output: Pass! \n";
        std::cout << std::endl;
    }
    {
        // sgemm_128x128_2buf_Divisible128 time: 6.16957 ms,    FP32 Perf: 22.2769 TFlops
        // sgemm_128x128_2buf              time: 8.06579 ms,    FP32 Perf: 17.0397 TFlops
        PROFILE((Sgemm128x128Buf2<<<grid2, block2>>>(mat_a_dev, mat_b_dev, mat_c_dev, m, k, n)),
                sgemm_128x128_2buf, m, k, n);

        CUDA_CHECK(cudaMemcpy(cuda_c.data(), mat_c_dev, m * n * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());

        CheckCudaOutput(cublas_c, cuda_c);
        std::cout << "Check Sgemm128x128Buf2 Output: Pass! \n";
        std::cout << std::endl;
    }

    CUDA_CHECK_AND_FREE(mat_a_dev);
    CUDA_CHECK_AND_FREE(mat_b_dev);
    CUDA_CHECK_AND_FREE(mat_c_dev);

    return 0;
}
