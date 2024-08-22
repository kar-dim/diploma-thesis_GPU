#pragma once
#include <cuda_runtime.h>

//Helper functions related to cuda
namespace cuda_utils {
    float* cudaMallocPtr(const std::size_t count);
    dim3 grid_size_calculate(const dim3 blockSize, const int rows, const int cols);
    cudaArray* cudaMallocArray(const std::size_t cols, const std::size_t rows);
    cudaResourceDesc createResourceDescriptor(cudaArray* cuArray);
    cudaTextureDesc createTextureDescriptor();
    cudaTextureObject_t createTextureObject(const cudaResourceDesc& pResDesc, const cudaTextureDesc& pTexDesc);
}
