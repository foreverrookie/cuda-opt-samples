add_executable(sgemm sgemm.cu)

target_link_libraries(sgemm
    PRIVATE
    CUDA::cudart
    CUDA::cublas
)

target_compile_options(sgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_compile_options(sgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
