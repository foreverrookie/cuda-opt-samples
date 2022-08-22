add_executable(sgemm sgemm.cu)

target_link_libraries(sgemm
    PRIVATE
    CUDA::cudart
    CUDA::cublas
)

set_property(TARGET sgemm
    PROPERTY
    CUDA_ARCHITECTURES 80
)
