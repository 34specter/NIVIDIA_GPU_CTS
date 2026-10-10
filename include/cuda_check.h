
#pragma once

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

inline void checkCuda(cudaError_t result, const char* call)
{
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(call) + " failed: " +
            cudaGetErrorString(result));
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call)