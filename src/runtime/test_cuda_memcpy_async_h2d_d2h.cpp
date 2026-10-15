#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstddef>

// 参数类型：size_t。资源属于夹具，断言提前结束后仍由 TearDown 清理。
class CudaMemcpyAsyncH2DD2HTest : public ::testing::TestWithParam<size_t> {
protected:
    void* input = nullptr;
    void* output = nullptr;
    void* device = nullptr;
    cudaStream_t stream = nullptr;
    void SetUp() override
    {
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        ASSERT_GT(count, 0) << "需要真实 NVIDIA GPU";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    }
    void TearDown() override
    {
        if (stream) {
            EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        }
        if (device) EXPECT_EQ(cudaFree(device), cudaSuccess);
        if (output) EXPECT_EQ(cudaFreeHost(output), cudaSuccess);
        if (input) EXPECT_EQ(cudaFreeHost(input), cudaSuccess);
    }
};

TEST_P(CudaMemcpyAsyncH2DD2HTest, CudaMemcpyAsyncH2DD2HCorrectness)
{
    const size_t count = GetParam();
    const size_t bytes = count * sizeof(int);
    ASSERT_EQ(cudaMallocHost(&input, bytes), cudaSuccess);
    ASSERT_EQ(cudaMallocHost(&output, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device, bytes), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    auto* in = static_cast<int*>(input);
    auto* out = static_cast<int*>(output);
    for (size_t i = 0; i < count; ++i) { in[i] = static_cast<int>(i + 5); out[i] = -1; }
    ASSERT_EQ(cudaMemcpyAsync(device, input, bytes, cudaMemcpyHostToDevice, stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(output, device, bytes, cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (size_t i = 0; i < count; ++i) EXPECT_EQ(out[i], in[i]) << "index=" << i;
    std::cout << "async_round_trip_elements=" << count << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    Parameters,
    CudaMemcpyAsyncH2DD2HTest,
    ::testing::Values(size_t{1}, size_t{257}, size_t{4096})
);
