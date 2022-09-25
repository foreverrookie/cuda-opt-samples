add_executable(reduce reduce.cu)

target_link_libraries(reduce
    PRIVATE
    CUDA::cudart
)
