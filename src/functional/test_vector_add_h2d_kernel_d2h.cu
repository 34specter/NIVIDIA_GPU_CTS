#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void vectorAdd(const float* a, const float* b, float* c, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) c[i] = a[i] + b[i];
}

class VectorAddH2DKernelD2HTest : public ::testing::TestWithParam<int> {
protected:
    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;

    void SetUp() override
    {
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        ASSERT_GT(count, 0) << "需要真实 NVIDIA GPU";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }

    void TearDown() override
    {
        if (device_a || device_b || device_c)
            EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        if (device_c) EXPECT_EQ(cudaFree(device_c), cudaSuccess);
        if (device_b) EXPECT_EQ(cudaFree(device_b), cudaSuccess);
        if (device_a) EXPECT_EQ(cudaFree(device_a), cudaSuccess);
    }
};

TEST_P(VectorAddH2DKernelD2HTest, GPUResultMatchesCPUReference)
{
    const int count = GetParam();
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    std::vector<float> a(count), b(count), output(count, -1.0f);
    for (int i = 0; i < count; ++i) {
        a[i] = static_cast<float>(i % 97) * 0.25f;
        b[i] = static_cast<float>(i % 31) * 0.5f;
    }

    ASSERT_EQ(cudaMalloc(&device_a, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_b, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_c, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    constexpr int block = 256;
    const int grid = (count + block - 1) / block;
    vectorAdd<<<grid, block>>>(device_a, device_b, device_c, count);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "kernel 启动失败";
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "kernel 执行失败";
    ASSERT_EQ(cudaMemcpy(output.data(), device_c, bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < count; ++i)
        EXPECT_NEAR(output[i], a[i] + b[i], 1e-6f) << "index=" << i;
    std::cout << "elements=" << count << ", grid=" << grid << ", block=" << block << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    ArrayLengths,
    VectorAddH2DKernelD2HTest,
    ::testing::Values(1, 255, 256, 257, 4097)
);
