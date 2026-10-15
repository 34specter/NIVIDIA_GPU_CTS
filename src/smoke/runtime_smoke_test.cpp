
#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <iostream>
#include <string>

#include "cuda_check.h"

struct DeviceVisibilityCase {
    const char* visible_devices;
    int expected_count;
};

class RuntimeSmokeTest
    : public ::testing::TestWithParam<DeviceVisibilityCase> {
protected:
    DeviceVisibilityCase test_case{};

    void SetUp() override
    {
        test_case = GetParam();

        // CUDA_VISIBLE_DEVICES must be set before the first CUDA Runtime API
        // call. In practice, different visibility configurations should be
        // run in separate processes because CUDA caches this configuration.
        ASSERT_EQ(setenv("CUDA_VISIBLE_DEVICES", test_case.visible_devices, 1), 0)
            << "Failed to set CUDA_VISIBLE_DEVICES";
    }

    void TearDown() override
    {
        // Release the resources created by the CUDA Runtime for this test.
        EXPECT_EQ(cudaDeviceReset(), cudaSuccess);
    }
};

TEST_P(RuntimeSmokeTest, EnumerateDevices)
{
    int actual_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&actual_count), cudaSuccess);

    EXPECT_EQ(actual_count, test_case.expected_count)
        << "CUDA_VISIBLE_DEVICES=" << test_case.visible_devices;

    std::cout << "CUDA_VISIBLE_DEVICES=" << test_case.visible_devices
              << ", expected_count=" << test_case.expected_count
              << ", actual_count=" << actual_count << '\n';
}

// Each visibility configuration is intended to be run in a fresh process.
// With gtest_discover_tests, keep one configuration here unless the test
// runner explicitly isolates parameterized cases into separate processes.
INSTANTIATE_TEST_SUITE_P(
    VisibleDevices,
    RuntimeSmokeTest,
    ::testing::Values(DeviceVisibilityCase{"0", 1}),
    [](const ::testing::TestParamInfo<DeviceVisibilityCase>& info) {
        return std::string("Device") + std::to_string(info.index);
    });

