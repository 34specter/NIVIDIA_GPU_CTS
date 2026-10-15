#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstddef>
#include <vector>
// 参数类型：size_t。资源属于夹具，断言提前结束后仍由 TearDown 清理。
class CudaMemcpyH2DD2HTest : public ::testing::TestWithParam<size_t> {
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

TEST_P(CudaMemcpyH2DD2HTest, CudaMemcpyH2DD2HCorrectness)
{
    const size_t count = GetParam();
    const size_t bytes = count * sizeof(int);
    std::vector<int> input(count), output(count, -1);
    for (size_t i = 0; i < count; ++i) input[i] = static_cast<int>(i * 3 + 7);
    ASSERT_EQ(cudaMalloc(&device, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device, input.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(output.data(), device, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < count; ++i) EXPECT_EQ(output[i], input[i]) << "index=" << i;
    std::cout << "round_trip_elements=" << count << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    Parameters,
    CudaMemcpyH2DD2HTest,
    ::testing::Values(size_t{1}, size_t{257}, size_t{4096})
);
