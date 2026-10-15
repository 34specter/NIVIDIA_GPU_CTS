#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstddef>
#include <vector>
// 参数类型：size_t。资源属于夹具，断言提前结束后仍由 TearDown 清理。
class CudaMemcpyD2DTest : public ::testing::TestWithParam<size_t> {
protected:
    void* source = nullptr;
    void* destination = nullptr;
    void SetUp() override
    {
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        ASSERT_GT(count, 0) << "需要真实 NVIDIA GPU";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    }
    void TearDown() override
    {
        if (destination) EXPECT_EQ(cudaFree(destination), cudaSuccess);
        if (source) EXPECT_EQ(cudaFree(source), cudaSuccess);
    }
};

TEST_P(CudaMemcpyD2DTest, CudaMemcpyD2DCorrectness)
{
    const size_t count = GetParam();
    const size_t bytes = count * sizeof(int);
    std::vector<int> input(count), output(count, -1);
    for (size_t i = 0; i < count; ++i) input[i] = static_cast<int>(i + 11);
    ASSERT_EQ(cudaMalloc(&source, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&destination, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(source, input.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(output.data(), destination, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < count; ++i) EXPECT_EQ(output[i], input[i]) << "index=" << i;
    std::cout << "d2d_elements=" << count << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    Parameters,
    CudaMemcpyD2DTest,
    ::testing::Values(size_t{1}, size_t{257}, size_t{4096})
);
