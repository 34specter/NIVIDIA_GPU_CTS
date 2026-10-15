#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstddef>

// 参数类型：size_t。资源属于夹具，断言提前结束后仍由 TearDown 清理。
class CudaMallocHostFreeHostTest : public ::testing::TestWithParam<size_t> {
protected:
    void* host = nullptr;
    void SetUp() override
    {
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        ASSERT_GT(count, 0) << "需要真实 NVIDIA GPU";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    }
    void TearDown() override
    {
        if (host) EXPECT_EQ(cudaFreeHost(host), cudaSuccess);
    }
};

TEST_P(CudaMallocHostFreeHostTest, CudaMallocHostFreeHostCorrectness)
{
    const size_t bytes = GetParam();
    ASSERT_EQ(cudaMallocHost(&host, bytes), cudaSuccess);
    ASSERT_NE(host, nullptr);
    auto* data = static_cast<unsigned char*>(host);
    for (size_t i = 0; i < bytes; ++i) data[i] = static_cast<unsigned char>(i % 251);
    for (size_t i = 0; i < bytes; ++i)
        EXPECT_EQ(data[i], static_cast<unsigned char>(i % 251));
    ASSERT_EQ(cudaFreeHost(host), cudaSuccess);
    host = nullptr;
    std::cout << "pinned_host_bytes=" << bytes << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    Parameters,
    CudaMallocHostFreeHostTest,
    ::testing::Values(size_t{1}, size_t{257}, size_t{4096})
);
