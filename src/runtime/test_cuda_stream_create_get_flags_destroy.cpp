#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <iostream>

// 参数类型：unsigned int。资源属于夹具，断言提前结束后仍由 TearDown 清理。
class CudaStreamFlagsTest : public ::testing::TestWithParam<unsigned int> {
protected:
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
        if (stream) EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
};

TEST_P(CudaStreamFlagsTest, CudaStreamCreateGetFlagsDestroyCorrectness)
{
    const unsigned int requested = GetParam();
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, requested), cudaSuccess);
    unsigned int actual = 0;
    ASSERT_EQ(cudaStreamGetFlags(stream, &actual), cudaSuccess);
    EXPECT_EQ(actual, requested);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    stream = nullptr;
    std::cout << "requested_flags=" << requested << ", actual_flags=" << actual << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    Parameters,
    CudaStreamFlagsTest,
    ::testing::Values(static_cast<unsigned int>(cudaStreamDefault), static_cast<unsigned int>(cudaStreamNonBlocking))
);
