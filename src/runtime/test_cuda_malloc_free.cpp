#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstddef>

// 参数类型：size_t。资源属于夹具，断言提前结束后仍由 TearDown 清理。
class CudaMallocFreeTest : public ::testing::TestWithParam<size_t> {
protected:
    void* device = nullptr;
    void SetUp() override
    {
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        ASSERT_GT(count, 0) << "需要真实 NVIDIA GPU";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    }
    void TearDown() override
    {
        if (device) EXPECT_EQ(cudaFree(device), cudaSuccess);
    }
};

TEST_P(CudaMallocFreeTest, CudaMallocFreeCorrectness)
{
    const size_t bytes = GetParam();
    ASSERT_EQ(cudaMalloc(&device, bytes), cudaSuccess);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(cudaFree(device), cudaSuccess);
    device = nullptr;
    std::cout << "allocated_and_freed_bytes=" << bytes << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    Parameters,
    CudaMallocFreeTest,
    ::testing::Values(size_t{1}, size_t{257}, size_t{4096})
);
