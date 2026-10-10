
#include <gtest/gtest.h>
#include <cuda.h>
#include <iostream>

#include "driver_check.h"

TEST(DriverSmokeTest, EnumerateDevices)
{
    DRIVER_CHECK(cuInit(0));

    int count = 0;
    DRIVER_CHECK(cuDeviceGetCount(&count));

    ASSERT_GT(count, 0) << "No NVIDIA GPU detected";

    CUdevice device;
    DRIVER_CHECK(cuDeviceGet(&device, 0));

    char name[256] = {};
    DRIVER_CHECK(cuDeviceGetName(name, sizeof(name), device));

    EXPECT_GT(name[0], '\0');

    std::cout << "GPU count: " << count << '\n';
    std::cout << "GPU name: " << name << '\n';
}