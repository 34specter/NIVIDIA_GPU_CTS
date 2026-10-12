#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <iostream>

// 测试参数：目标 GPU 编号。
class CudaSetGetDeviceTest : public ::testing::TestWithParam<int> {
};

TEST_P(CudaSetGetDeviceTest, SelectedDeviceMatchesRequestedDevice)
{
    const int target_device = GetParam();

    // 1. 准备：查询设备数量，检查目标设备存在。
    int device_count = 0;

    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    ASSERT_GE(target_device, 0);
    ASSERT_LT(target_device, device_count)
        << "目标设备不存在，实际设备数量：" << device_count;

    // 2. 调用：选择目标设备。
    ASSERT_EQ(cudaSetDevice(target_device), cudaSuccess);

    // 3. 调用：查询当前设备。
    int current_device = -1;

    ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);

    // 4. 检查输出。
    EXPECT_EQ(current_device, target_device);

    // 5. 记录日志。
    std::cout << "device_count=" << device_count
              << ", target_device=" << target_device
              << ", current_device=" << current_device
              << '\n';
}

// 第一阶段只测试 GPU 0。
INSTANTIATE_TEST_SUITE_P(
    GPU0,
    CudaSetGetDeviceTest,
    ::testing::Values(0)
);