
#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <iostream>

#include "cuda_check.h"

TEST(RuntimeSmokeTest, EnumerateDevices)
{
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));

    ASSERT_GT(count, 0) << "No NVIDIA GPU detected";

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    EXPECT_GT(prop.totalGlobalMem, size_t{0});
    EXPECT_GT(prop.multiProcessorCount, 0);

    std::cout << "GPU count: " << count << '\n';
    std::cout << "GPU name: " << prop.name << '\n';
    std::cout << "Global memory: "
              << prop.totalGlobalMem / (1024 * 1024)
              << " MiB\n";
}