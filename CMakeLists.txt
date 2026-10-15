cmake_minimum_required(VERSION 3.18)

project(nvidia_gpu_cts LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

enable_testing()
include(GoogleTest)

add_executable(driver_smoke_test src/smoke/driver_smoke_test.cpp)
target_include_directories(driver_smoke_test PRIVATE include)
target_link_libraries(driver_smoke_test PRIVATE CUDA::cuda_driver GTest::gtest_main)
gtest_discover_tests(driver_smoke_test DISCOVERY_MODE PRE_TEST)

add_executable(runtime_smoke_test src/smoke/runtime_smoke_test.cpp)
target_include_directories(runtime_smoke_test PRIVATE include)
target_link_libraries(runtime_smoke_test PRIVATE CUDA::cudart GTest::gtest_main)
gtest_discover_tests(runtime_smoke_test DISCOVERY_MODE PRE_TEST)

# 每个名称对应同名 .cpp 和同名独立测试程序。
set(RUNTIME_TESTS
    test_cuda_set_get_device
    test_cuda_get_device_properties
    test_cuda_malloc_free
    test_cuda_memcpy_h2d_d2h
    test_cuda_memcpy_d2d
    test_cuda_memset
    test_cuda_mem_get_info
    test_cuda_malloc_host_free_host
    test_cuda_memcpy_async_h2d_d2h
    test_cuda_stream_create_get_flags_destroy
    test_cuda_event_create_destroy
    test_cuda_event_record_synchronize_query
    test_cuda_event_elapsed_time
    test_cuda_set_device_invalid_ordinal
    test_cuda_malloc_pitch_memcpy2d
    test_cuda_malloc_managed_memset
)

foreach(test_name IN LISTS RUNTIME_TESTS)
    add_executable(${test_name} src/runtime/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE CUDA::cudart GTest::gtest_main)
    gtest_discover_tests(${test_name} DISCOVERY_MODE PRE_TEST)
endforeach()

add_executable(test_vector_add_h2d_kernel_d2h
    src/functional/test_vector_add_h2d_kernel_d2h.cu
)
target_link_libraries(test_vector_add_h2d_kernel_d2h PRIVATE CUDA::cudart GTest::gtest_main)
gtest_discover_tests(test_vector_add_h2d_kernel_d2h DISCOVERY_MODE PRE_TEST)
