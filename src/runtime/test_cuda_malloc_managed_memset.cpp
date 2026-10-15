#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstddef>

// 参数类型：size_t。资源属于夹具，断言提前结束后仍由 TearDown 清理。
class CudaMallocManagedMemsetTest : public ::testing::TestWithParam<size_t> {
protected:
    void* device = nullptr;
    void SetUp() override
    {
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        ASSERT_GT(count, 0) << "需要真实 NVIDIA GPU";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        int supported = 0;
        ASSERT_EQ(cudaDeviceGetAttribute(&supported, cudaDevAttrManagedMemory, 0), cudaSuccess);
        if (!supported) GTEST_SKIP() << "设备不支持 Unified Memory";
    }
    void TearDown() override
    {
        if (device) {
            EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            EXPECT_EQ(cudaFree(device), cudaSuccess);
        }
    }
};

TEST_P(CudaMallocManagedMemsetTest, CudaMallocManagedMemsetCorrectness)
{
    const size_t bytes = GetParam();
    ASSERT_EQ(cudaMallocManaged(&device, bytes), cudaSuccess);
    auto* data = static_cast<unsigned char*>(device);
    for (size_t i = 0; i < bytes; ++i) data[i] = 0;
    ASSERT_EQ(cudaMemset(device, 0x3c, bytes), cudaSuccess);
    // GPU 操作完成后，CPU 才读取托管内存。
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (size_t i = 0; i < bytes; ++i) EXPECT_EQ(data[i], 0x3c) << "index=" << i;
    std::cout << "managed_bytes=" << bytes << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    Parameters,
    CudaMallocManagedMemsetTest,
    ::testing::Values(size_t{1}, size_t{257}, size_t{4096})
);
