#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//constant array used for optimizing share memory accesses for Rx
//Helps with reducing the local memory required for each block for Rx arrays from 4096 to 2304
__constant__ int Rx_mappings[64] = {
        0,  1,  2,  3,  4,  5,  6,  7,
        1,  8,  9,  10, 11, 12, 13, 14,
        2,  9,  15, 16, 17, 18, 19, 20,
        3,  10, 16, 21, 22, 23, 24, 25,
        4,  11, 17, 22, 26, 27, 28, 29,
        5,  12, 18, 23, 27, 30, 31, 32,
        6,  13, 19, 24, 28, 31, 33, 34,
        7,  14, 20, 25, 29, 32, 34, 35
};
__global__ void nvf(cudaTextureObject_t texObj, float* m_nvf, const int p_squared, const int pad, const int width, const int height);
__global__ void me_p3(cudaTextureObject_t texObj, float* Rx, float* rx, const int width, const int padded_width, const int height);