
#pragma once

#include <cuda.h>
#include <stdexcept>
#include <string>

inline void checkDriver(CUresult result, const char* call)
{
    if (result == CUDA_SUCCESS) {
        return;
    }

    const char* error = nullptr;
    cuGetErrorName(result, &error);

    throw std::runtime_error(
        std::string(call) + " failed: " +
        (error ? error : "unknown error"));
}

#define DRIVER_CHECK(call) checkDriver((call), #call)