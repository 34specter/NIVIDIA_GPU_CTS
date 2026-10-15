# NVIDIA GPU CTS：15 类 Runtime API + 向量加法功能测试

基于仓库提交 `a75a36e` 增加代码。保留两个 smoke 程序和你已写的
`test_cuda_set_get_device.cpp`。新代码为 15 个 .cpp、1 个 .cu，逐文件对应独立可执行程序。
全部使用真实 CUDA API，无模拟实现。尚未执行 GPU 实测，也未验证真实 SDK 下的编译。

## 依赖

- Linux、真实 NVIDIA GPU、兼容驱动和 CUDA Toolkit（含 nvcc）
- CMake >= 3.18，支持 C++17/CUDA17 的编译器
- 可被 find_package(GTest REQUIRED) 找到的 GoogleTest，建议 >= 1.10

## 编译

在项目根目录执行；CUDA 12.8 路径仅为例子，需改成实际路径。

```bash
cmake -S . -B build-runtime15 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build build-runtime15 -j
```

CMake 已启用 CUDA 语言以构建 .cu 文件，使用 CUDA::cudart 和 CUDA::cuda_driver，
不再硬编码库文件路径。使用新 build 目录避免继承旧的 stub/RPATH 配置。
如需设置 GPU 架构，可在首次配置时追加 `-DCMAKE_CUDA_ARCHITECTURES=实际SM编号`。

## 从最简单的用例开始运行

```bash
./build-runtime15/test_cuda_malloc_free
./build-runtime15/test_cuda_memcpy_h2d_d2h
./build-runtime15/test_vector_add_h2d_kernel_d2h
```

查看参数化测试名称并导出 XML：

```bash
./build-runtime15/test_cuda_memcpy_h2d_d2h --gtest_list_tests
mkdir -p reports
./build-runtime15/test_cuda_memcpy_h2d_d2h \
  --gtest_output=xml:reports/test_cuda_memcpy_h2d_d2h.xml
```

CTest 全量执行（含原有用例），先查看列表再执行：

```bash
ctest --test-dir build-runtime15 -N
ctest --test-dir build-runtime15 --output-on-failure --timeout 120
```

默认选择可见设备 0。如需指定一张物理 GPU，运行前设置
`CUDA_VISIBLE_DEVICES=GPU-实际UUID`；程序中的 ordinal 0 将对应这张卡。
没有 GPU/驱动初始化失败是环境失败，不作为通过。托管内存能力不支持时显式 skip。

## 新增用例

15 类 Runtime API 共 38 个参数实例；功能测试含 5 个长度实例。
加上原有 2 个 smoke 和设备选择实例，共预期 46 个 GoogleTest 实例。
这里“15 条”指 15 个接口测试场景，不把参数实例当作 15 个独立设计场景。

| 文件（位于 src/runtime，后缀 .cpp） | 验证内容 |
|---|---|
| `test_cuda_get_device_properties` | 设备属性：名称、显存、SM 和线程上限 |
| `test_cuda_malloc_free` | 显存分配、非空指针、显式释放 |
| `test_cuda_memcpy_h2d_d2h` | CPU→GPU→CPU 数据一致性 |
| `test_cuda_memcpy_d2d` | GPU→GPU 拷贝后回传校验 |
| `test_cuda_memset` | 0x00、0x7f、0xff 字节填充 |
| `test_cuda_mem_get_info` | 查询空闲和总显存，验证 free <= total |
| `test_cuda_malloc_host_free_host` | 页锁定主机内存分配、读写与释放 |
| `test_cuda_memcpy_async_h2d_d2h` | 页锁定内存异步往返拷贝，流同步后校验 |
| `test_cuda_stream_create_get_flags_destroy` | 流创建、标志查询、销毁 |
| `test_cuda_event_create_destroy` | 不同标志的事件创建与销毁 |
| `test_cuda_event_record_synchronize_query` | 事件记录、同步、完成查询及输出校验 |
| `test_cuda_event_elapsed_time` | 事件计时返回有效非负耗时（非性能门禁） |
| `test_cuda_set_device_invalid_ordinal` | 越界设备编号错误返回及有效调用恢复 |
| `test_cuda_malloc_pitch_memcpy2d` | 带 pitch 二维内存分配和往返数据校验 |
| `test_cuda_malloc_managed_memset` | 托管内存支持探测、GPU填充、同步后CPU读取 |

功能文件：`src/functional/test_vector_add_h2d_kernel_d2h.cu`

主流程：准备 A/B → 分配显存 → H2D → vectorAdd kernel → 启动错误检查 →
设备同步 → D2H → CPU 参考比较。N=1/255/256/257/4097，block=256，
验证整块和非整块边界。CPU 输入是已知测试数据，GPU 输出由真实 kernel 计算。

## 代码结构

每个文件都采用 TestWithParam → SetUp → TEST_P → TearDown → Values。
主业务流程写在 TEST_P，不引入额外执行框架。资源句柄以 nullptr 初始化，
TearDown 根据有效句柄清理；即使 ASSERT 提前结束也执行清理。
异步操作在读取主机输出、释放主机或设备缓冲区前同步。

## 后续 pytest 接入

CSV 选择可执行程序、gtest_filter 和超时。pytest 收集 stdout/stderr、退出码和 XML，
必须检查实际执行数量、失败与 skipped，避免“0 个用例但退出码 0”的假通过。
本包没有实现 pytest、Allure 或 Jenkins，不声称自动化链路已完成。

## 已完成的验证与限制

通过：新增源码数量、独立目标映射、参数实例数量、资源清理结构、功能 kernel
边界判断、启动与执行错误检查、数据比较结构、ZIP 完整性检查。
当前编写环境无 CMake、GoogleTest、CUDA Toolkit/nvcc 和 NVIDIA GPU，
因此未完成编译、链接、CTest 发现及 GPU 运行验证。请以 GPU 服务器输出为准。

## 参考

- https://docs.nvidia.com/cuda/archive/12.8.2/cuda-runtime-api/index.html
- https://github.com/NVIDIA/cuda-samples （vectorAdd 核心计算思路，显式拷贝按本项目设计）
- https://cmake.org/cmake/help/v3.24/module/FindCUDAToolkit.html
