#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstddef>

// 参数类型：int。资源属于夹具，断言提前结束后仍由 TearDown 清理。
class CudaMemGetInfoTest : public ::testing::TestWithParam<int> {
protected:

    void SetUp() override
    {
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        ASSERT_GT(count, 0) << "需要真实 NVIDIA GPU";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    }
};

TEST_P(CudaMemGetInfoTest, CudaMemGetInfoCorrectness)
{
    ASSERT_EQ(cudaSetDevice(GetParam()), cudaSuccess);
    size_t free_bytes = 0, total_bytes = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);
    EXPECT_GT(total_bytes, size_t{0});
    EXPECT_LE(free_bytes, total_bytes);
    // 不断言空闲显存等于某个固定值，共享 GPU 上它会变化。
    std::cout << "free_bytes=" << free_bytes << ", total_bytes=" << total_bytes << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    Parameters,
    CudaMemGetInfoTest,
    ::testing::Values(0)
);
