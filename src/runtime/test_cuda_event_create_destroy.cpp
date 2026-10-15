#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <iostream>

// 参数类型：unsigned int。资源属于夹具，断言提前结束后仍由 TearDown 清理。
class CudaEventCreateDestroyTest : public ::testing::TestWithParam<unsigned int> {
protected:
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
        if (event) EXPECT_EQ(cudaEventDestroy(event), cudaSuccess);
    }
};

TEST_P(CudaEventCreateDestroyTest, CudaEventCreateDestroyCorrectness)
{
    const unsigned int flags = GetParam();
    ASSERT_EQ(cudaEventCreateWithFlags(&event, flags), cudaSuccess);
    ASSERT_NE(event, nullptr);
    ASSERT_EQ(cudaEventDestroy(event), cudaSuccess);
    event = nullptr;
    std::cout << "event_flags=" << flags << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    Parameters,
    CudaEventCreateDestroyTest,
    ::testing::Values(static_cast<unsigned int>(cudaEventDefault), static_cast<unsigned int>(cudaEventBlockingSync), static_cast<unsigned int>(cudaEventDisableTiming))
);
