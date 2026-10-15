#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstddef>
#include <vector>
// 参数类型：size_t。资源属于夹具，断言提前结束后仍由 TearDown 清理。
class CudaMallocPitchMemcpy2DTest : public ::testing::TestWithParam<size_t> {
protected:
    void* device = nullptr;
    void SetUp() override
    {
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        ASSERT_GT(count, 0) << "需要真实 NVIDIA GPU";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    }
    void TearDown() override
    {
        if (device) EXPECT_EQ(cudaFree(device), cudaSuccess);
    }
};

TEST_P(CudaMallocPitchMemcpy2DTest, CudaMallocPitchMemcpy2DCorrectness)
{
    const size_t columns = GetParam();
    constexpr size_t rows = 7;
    const size_t row_bytes = columns * sizeof(int);
    std::vector<int> input(rows * columns), output(rows * columns, -1);
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<int>(i + 3);
    size_t pitch = 0;
    ASSERT_EQ(cudaMallocPitch(&device, &pitch, row_bytes, rows), cudaSuccess);
    ASSERT_GE(pitch, row_bytes);
    ASSERT_EQ(cudaMemcpy2D(device, pitch, input.data(), row_bytes,
                          row_bytes, rows, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy2D(output.data(), row_bytes, device, pitch,
                          row_bytes, rows, cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < input.size(); ++i) EXPECT_EQ(output[i], input[i]) << "index=" << i;
    std::cout << "columns=" << columns << ", rows=" << rows << ", pitch_bytes=" << pitch << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    Parameters,
    CudaMallocPitchMemcpy2DTest,
    ::testing::Values(size_t{1}, size_t{17}, size_t{257})
);
