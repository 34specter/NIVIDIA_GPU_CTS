cmake_minimum_required(VERSION 3.18)
project(nvidia_gpu_cts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA paths
set(CUDA_INCLUDE_PATH "/usr/local/cuda/include")
set(CUDA_RUNTIME_LIB  "/usr/local/cuda/lib64/libcudart.so")
set(CUDA_DRIVER_LIB   "/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs/libcuda.so")

find_package(GTest REQUIRED)

enable_testing()

add_executable(driver_smoke_test
    src/smoke/driver_smoke_test.cpp
)

target_include_directories(driver_smoke_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_PATH}
)

target_link_libraries(driver_smoke_test PRIVATE
    ${CUDA_DRIVER_LIB}
    GTest::gtest_main
)

add_executable(runtime_smoke_test
    src/smoke/runtime_smoke_test.cpp
)

target_include_directories(runtime_smoke_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_PATH}
)

target_link_libraries(runtime_smoke_test PRIVATE
    ${CUDA_RUNTIME_LIB}
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(driver_smoke_test)
gtest_discover_tests(runtime_smoke_test)