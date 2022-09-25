# CUDA Optimization Samples

CUDA Optimization Samples including Sgemm(Single precision general Matrix Multiply), reduce...
To be continued.

## build

```bash
cmake . -B build
cmake --build build
```

> Note: If `CMAKE_CUDA_ARCHITECTURES` is not assigned, 86 is adopted.

run sgemm

```bash
./build/sgemm/sgemm [m] [k] [n]
```

## Sgemm

My test platform is Nvidia **3080** GPU which has FP32 peak performance of **29.8** TFlops.

Test matrix size is 4096(m), 4096(k), 4096(n).

| Kernel             | Time(ms)| FP32 Performance(TFlops) | Relative(cuBLAS) Performance |
| ---                | ---     | ---     | ---      |
| cublasSgemm        | 7.19 | 19.12 |  1       |
| SgemmNaive         | 72.76 | 1.89 | 0.099 |
| SgemmNaive2        | 60.84 | 2.26 | 0.12 |
| SgemmNaive4        | 54.82 | 2.51 | 0.13 |
| sgemm_16x16        | 50.42 | 2.72 | 0.14 |
| sgemm_128x128      | 9.25 | 14.86 | 0.78 |
| sgemm_128x128_2buf | 8.07 | 17.04 | 0.89 |
| sgemm_128x128_2buf_opt | 7.78 | 17.68 | 0.92 |

1. `cublasSgemm` attains **19.1221** TFlops. Note: In cuBLAS, matrix A, B and C are stored by column-major.
If row-major output wanted, convert to CT = BT * AT. Don't Transpose input matrix, so we can get a fair kernel time(pure matrix multiply).

2. Kernel `SgemmNaive` traverses Dim-K, and only get **1.88882** TFlops Performance.
Because it has a large number of redundant global load(low Arithmetic Intensity).
Then test `SgemmNaive2` and `SgemmNaive4`, attain **2.25891** TFlops and **2.50723** TFlops respectively.
`FETCH_FLOAT2` & `FETCH_FLOAT4` is reference to https://github.com/Liu-xiandong/How_to_optimize_in_GPU/

3. Kernel `Sgemm16x16` is based on `Sgemm16x16Divisible16` which assume input size(m,k,n) is a multiple of 16.

4. Kernel `Sgemm128x128` is based on `Sgemm128x128Divisible128` which assume input size(m,k,n) is a multiple of 128.
Here, I adopt config in [CUTLASS](https://github.com/NVIDIA/cutlass). 

    |    Item           | Detail |
    |   ---             |  ---   |
    | Block size          | 256    |
    | Thread computation granularity | 8 x 8 (4 x 4, 2x2 blocks)   |
    | Block computation granularity |  128 x 128        |
    | Warp Shape        | 8 x 4 |

5. Kernel `Sgemm128x128Buf2` is based on `Sgemm128x128Buf2Divisible128` which assume input size(m,k,n) is a multiple of 128. It applys double buffers in shared memory & register files, and gets 17.0397 TFlops(89% to cuBLAS) performance.
    - Further more, changing thread mapping coded in kernel `Sgemm128x128Buf2Opt` gets 17.6795 TFlops(92% to cuBLAS)

6. Further optimization based on kernel `Sgemm128x128Buf2Opt` is coming soon.

## Reduce

Memory-Bound.
Nvidia **3080** GPU has GDDR6X bandwidth of **760GB/s**.

Test array length is 104857600(100 x 1024 x 1024).

| Kernel                            | Time(ms)  | Bandwidth(GB/s)   | Relative(theoretical) Bandwidth |
| ---                               | ---       | ---               | --- |
| reduce_naive                      | 171.08    | 2.28  | 0.003 |
| reduce_shm                        | 1.55      | 251.6 | 0.33  |
| reduce_thread_loop                | 0.587     | 665.7 | 0.8759 |
| reduce_thread_loop_warp_unroll    | 0.586     | 666.9 | 0.8775 |

1. Each thread in kernel `ReduceNaive` adopts `atomicAdd` to dst. So it only has 0.3% Peak Bandwidth.

2. Using shared memory to accumulate in current block(`ReduceShm`) increases the bandwidth to 251.6 GB/s.

3. Kernel `ReduceThreadLoop` increases thread computation granularity(default 8), and gets 665.7 GB/s.

4. Kernel `ReduceThreadLoopWarpUnroll` [unrolls last warp](https://developer.download.nvidia.cn/assets/cuda/files/reduction.pdf), but little effect.
