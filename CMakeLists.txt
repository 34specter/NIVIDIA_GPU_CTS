cmake_minimum_required(VERSION 3.18)
project(nvidia_gpu_cts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

enable_testing()

add_executable(driver_smoke_test
    src/smoke/driver_smoke_test.cpp
)

target_include_directories(driver_smoke_test PRIVATE include)
target_link_libraries(driver_smoke_test
    PRIVATE CUDA::cuda_driver GTest::gtest_main
)

add_executable(runtime_smoke_test
    src/smoke/runtime_smoke_test.cpp
)

target_include_directories(runtime_smoke_test PRIVATE include)
target_link_libraries(runtime_smoke_test
    PRIVATE CUDA::cudart GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(driver_smoke_test)
gtest_discover_tests(runtime_smoke_test)