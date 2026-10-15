#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstddef>
#include <vector>
// 参数类型：size_t。资源属于夹具，断言提前结束后仍由 TearDown 清理。
class CudaEventRecordSynchronizeQueryTest : public ::testing::TestWithParam<size_t> {
protected:
    void* device = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
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
        if (event) EXPECT_EQ(cudaEventDestroy(event), cudaSuccess);
        if (stream) EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        if (device) EXPECT_EQ(cudaFree(device), cudaSuccess);
    }
};

TEST_P(CudaEventRecordSynchronizeQueryTest, CudaEventRecordSynchronizeQueryCorrectness)
{
    const size_t bytes = GetParam();
    std::vector<unsigned char> output(bytes);
    ASSERT_EQ(cudaMalloc(&device, bytes), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&event), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(device, 0x5a, bytes, stream), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(event, stream), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(event), cudaSuccess);
    EXPECT_EQ(cudaEventQuery(event), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(output.data(), device, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < bytes; ++i) EXPECT_EQ(output[i], 0x5a) << "index=" << i;
    std::cout << "event_completed_bytes=" << bytes << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    Parameters,
    CudaEventRecordSynchronizeQueryTest,
    ::testing::Values(size_t{1}, size_t{257}, size_t{4096})
);
