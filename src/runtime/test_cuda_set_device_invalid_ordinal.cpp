#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <iostream>

// 参数类型：int。资源属于夹具，断言提前结束后仍由 TearDown 清理。
class CudaSetDeviceInvalidOrdinalTest : public ::testing::TestWithParam<int> {
protected:

    void SetUp() override
    {
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        ASSERT_GT(count, 0) << "需要真实 NVIDIA GPU";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    }
};

TEST_P(CudaSetDeviceInvalidOrdinalTest, CudaSetDeviceInvalidOrdinalCorrectness)
{
    int count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    const int invalid_ordinal = count + GetParam();
    EXPECT_EQ(cudaSetDevice(invalid_ordinal), cudaErrorInvalidDevice);
    // 清除本线程错误状态，再确认有效调用可恢复。
    (void)cudaGetLastError();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    int current = -1;
    ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, 0);
    std::cout << "invalid_ordinal=" << invalid_ordinal << ", recovered_device=" << current << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    Parameters,
    CudaSetDeviceInvalidOrdinalTest,
    ::testing::Values(0, 1)
);
