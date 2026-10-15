#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstddef>
#include <cmath>
// 参数类型：size_t。资源属于夹具，断言提前结束后仍由 TearDown 清理。
class CudaEventElapsedTimeTest : public ::testing::TestWithParam<size_t> {
protected:
    void* device = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    void SetUp() override
    {
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        ASSERT_GT(count, 0) << "需要真实 NVIDIA GPU";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    }
    void TearDown() override
    {
        if (stream) EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        if (stop) EXPECT_EQ(cudaEventDestroy(stop), cudaSuccess);
        if (start) EXPECT_EQ(cudaEventDestroy(start), cudaSuccess);
        if (stream) EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        if (device) EXPECT_EQ(cudaFree(device), cudaSuccess);
    }
};

TEST_P(CudaEventElapsedTimeTest, CudaEventElapsedTimeCorrectness)
{
    const size_t bytes = GetParam();
    ASSERT_EQ(cudaMalloc(&device, bytes), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(start, stream), cudaSuccess);
    for (int i = 0; i < 10; ++i)
        ASSERT_EQ(cudaMemsetAsync(device, i, bytes, stream), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(stop, stream), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);
    float elapsed_ms = -1.0f;
    ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, start, stop), cudaSuccess);
    EXPECT_TRUE(std::isfinite(elapsed_ms));
    EXPECT_GE(elapsed_ms, 0.0f);
    // 仅检查计时 API；没有硬编码性能门槛。
    std::cout << "bytes=" << bytes << ", elapsed_ms=" << elapsed_ms << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    Parameters,
    CudaEventElapsedTimeTest,
    ::testing::Values(size_t{4096}, size_t{1048576})
);
