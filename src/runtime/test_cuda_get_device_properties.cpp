#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstddef>

// 参数类型：int。资源属于夹具，断言提前结束后仍由 TearDown 清理。
class CudaGetDevicePropertiesTest : public ::testing::TestWithParam<int> {
protected:

    void SetUp() override
    {
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        ASSERT_GT(count, 0) << "需要真实 NVIDIA GPU";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    }
};

TEST_P(CudaGetDevicePropertiesTest, CudaGetDevicePropertiesCorrectness)
{
    const int ordinal = GetParam();
    cudaDeviceProp prop{};
    ASSERT_EQ(cudaGetDeviceProperties(&prop, ordinal), cudaSuccess);
    EXPECT_NE(prop.name[0], '\0');
    EXPECT_GT(prop.totalGlobalMem, size_t{0});
    EXPECT_GT(prop.multiProcessorCount, 0);
    EXPECT_GT(prop.maxThreadsPerBlock, 0);
    std::cout << "device=" << ordinal << ", name=" << prop.name
              << ", total_bytes=" << prop.totalGlobalMem << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    Parameters,
    CudaGetDevicePropertiesTest,
    ::testing::Values(0)
);
