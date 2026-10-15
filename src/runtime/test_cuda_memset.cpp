#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstddef>
#include <vector>
// 参数类型：int。资源属于夹具，断言提前结束后仍由 TearDown 清理。
class CudaMemsetTest : public ::testing::TestWithParam<int> {
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

TEST_P(CudaMemsetTest, CudaMemsetCorrectness)
{
    const int value = GetParam();
    constexpr size_t bytes = 257;
    std::vector<unsigned char> output(bytes);
    ASSERT_EQ(cudaMalloc(&device, bytes), cudaSuccess);
    // cudaMemset 填充字节，不是给 int/float 数组赋相同数值。
    ASSERT_EQ(cudaMemset(device, value, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(output.data(), device, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < bytes; ++i)
        EXPECT_EQ(output[i], static_cast<unsigned char>(value)) << "index=" << i;
    std::cout << "byte_pattern=" << value << ", bytes=" << bytes << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    Parameters,
    CudaMemsetTest,
    ::testing::Values(0, 0x7f, 0xff)
);
