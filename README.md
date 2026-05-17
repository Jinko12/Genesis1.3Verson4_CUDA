# Genesis 1.3 CUDA GPU Resident

**版本：** `GPUResident Stage 4.2B
**上游基线：** [Genesis-1.3-Version4](https://github.com/svenreiche/Genesis-1.3-Version4) (4.6.12)

本仓库在 Sven Reiche 的 Genesis 1.3 Version 4 之上加入 CUDA GPU 加速路径，目标是让 **beam、field、source deposition、field solve、diagnostics 和 slippage 尽可能保持 GPU resident**，只在 HDF5 输出、CPU fallback、必要 MPI 边界或显式调试时才同步到 host。

**可用一键开关 `GENESIS_CUDA_SAFE_MODE=1` 关闭所有可选 fast path。

> 上游版权与许可证遵循 Sven Reiche 原始仓库，本仓库只对 CUDA 路径与运行配套工具做改造与补充。

---

## 目录

- [1. 版本定位与核心原则](#1-版本定位与核心原则)
- [2. 推荐硬件与软件环境](#2-推荐硬件与软件环境)
- [3. 编译](#3-编译)
- [4. 输入文件配置](#4-输入文件配置)
- [5. 默认内置的 CUDA 行为](#5-默认内置的-cuda-行为)
- [6. 基本运行方式](#6-基本运行方式)
- [7. MPS 推荐运行](#7-mps-推荐运行)
- [8. 已验证性能摘要](#11-已验证性能摘要)
- [9. 已知限制与 FAQ](#12-已知限制与-faq)

---

## 1. 版本定位与核心原则

- 在保持 Genesis 原有物理模型和输入文件语义的基础上加入 GPU resident 主路径。
- 已验证稳定的 CUDA 优化路径默认开启。
- 保留所有 CPU fallback 与安全模式开关。
- **不会自动替用户改变 solver 选择**：是否启用 CUDA FFT field solver，仍由输入文件中的 `cuda_fieldsolver=true` / `fft_fieldsolver=true` 决定。
- **不会自动启动 MPS**：MPS 是运行环境策略，由用户或作业脚本控制。

---

## 2. 推荐硬件与软件环境

### 2.1 硬件

- NVIDIA GPU：建议 A100、H100、L40S、RTX 6000 Ada、RTX 4090 或同级。
- 显存：建议 ≥ 24 GB；大规模 slices/particles/多 field 需更多。
- CPU：≥ 16 核。
- 内存：≥ 64 GB。

### 2.2 软件

- Linux x86_64
- CMake 3.18+
- 支持 C++17 的 GCC/G++
- CUDA Toolkit（NVCC、cuFFT、cuDART）
- MPI：OpenMPI 或 MPICH
- HDF5：建议 parallel HDF5
- FFTW（可选，用于 CPU FFT fallback 或 CPU 对照）

---

## 3. 编译

```bash
cmake -S . -B build-cuda \
  -DUSE_CUDA=ON \
  -DGENESIS_CUDA_ARCHITECTURES=80

cmake --build build-cuda -j
```

按 GPU 架构调整 `GENESIS_CUDA_ARCHITECTURES`：

| GPU | 推荐值 |
|---|---:|
| V100 | 70 |
| RTX 20 系 / T4 | 75 |
| A100 / A30 | 80 |
| RTX 30 系 | 86 |
| L40S / RTX 6000 Ada / RTX 40 系 | 89 |
| H100 | 90 |

如果 HDF5 在 conda 环境中：

```bash
cmake -S . -B build-cuda \
  -DUSE_CUDA=ON \
  -DGENESIS_CUDA_ARCHITECTURES=80 \
  -DHDF5_ROOT=/path/to/conda/env
```

编译产物：`build-cuda/genesis4`

### CPU 对照编译（可选）

```bash
cmake -S . -B build-cpu -DUSE_CUDA=OFF
cmake --build build-cpu -j
```

> 严格性能对照时应区分 CPU ADI、CPU FFTW、CUDA FFT，不建议把 CPU ADI 与 CUDA FFT 作为 solver-to-solver 的严格对照。

---

## 4. 输入文件配置

在 `&track` 中启用 CUDA FFT field solver：

```text
&track
  fft_fieldsolver = true
  cuda_fieldsolver = true
&end
```

`cuda_beam` 默认跟随 `cuda_fieldsolver`，通常不需要单独设置。

仓库内提供的示例输入：

```
examples/Example3-TimeDependent/Example3.cuda.in
examples/Example4-HGHG/Example4_a.cuda.in
examples/Example4-HGHG/Example4_a.cuda_profile.in   # profiling 专用，lattice 路径用相对仓库根目录
```

---

## 5. 默认内置的 CUDA 行为

该版本已经把下列优化的默认值内置到程序中，用户**不再需要手动设置**：

```bash
GENESIS_CUDA_DEVICE_POLICY=local_rank
GENESIS_CUDA_FAST_KERNELS=1
GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD=1
GENESIS_CUDA_INPLACE_SLIPPAGE=1
GENESIS_CUDA_LONGITUDINAL_ALGEBRA_OPT=1
GENESIS_CUDA_BIND_FFT_FIELD=1
GENESIS_CUDA_DEFER_FIELD_D2H=1
GENESIS_CUDA_MPI_SLIPPAGE=1
GENESIS_CUDA_DIAG_REDUCTION=1
```

含义概览：

| 默认项 | 默认值 | 作用 |
|---|---:|---|
| `GENESIS_CUDA_DEVICE_POLICY` | `local_rank` | MPI local rank 按可见 GPU 轮转映射 |
| `GENESIS_CUDA_FAST_KERNELS` | 1 | 启用已验证的 CUDA fast kernels |
| `GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD` | 1 | longitudinal RK 步内缓存 field interpolation |
| `GENESIS_CUDA_INPLACE_SLIPPAGE` | 1 | 使用 in-place CUDA slippage，避免 scratch D2D copy |
| `GENESIS_CUDA_LONGITUDINAL_ALGEBRA_OPT` | 1 | FP64 代数等价整理 |
| `GENESIS_CUDA_BIND_FFT_FIELD` | 1 | BeamSolver 直接绑定 FFT field device buffer |
| `GENESIS_CUDA_DEFER_FIELD_D2H` | 1 | 默认延迟 field 的 D2H，只在必要边界同步 |
| `GENESIS_CUDA_MPI_SLIPPAGE` | 1 | multi-rank slippage 默认使用 CUDA-resident path |
| `GENESIS_CUDA_DIAG_REDUCTION` | 1 | diagnostics 默认 GPU compact reduction |

每个变量都仍可显式设为 0 强制回退。

---

## 6. 基本运行方式

### 6.1 单 GPU

```bash
CUDA_VISIBLE_DEVICES=0 \
mpirun -np 8 ./build-cuda/genesis4 examples/Example4-HGHG/Example4_a.cuda_profile.in
```

经验上单 GPU 不是 rank 越多越快，建议先 sweep `2, 4, 8, 12 ranks/GPU`：

```bash
CUDA_VISIBLE_DEVICES=0 \
tools/cuda_stage3_9_worker_sweep.sh \
  ./build-cuda/genesis4 \
  examples/Example4-HGHG/Example4_a.cuda_profile.in \
  1,2,4,8,12,16,24,32
```

### 6.2 多 GPU

```bash
CUDA_VISIBLE_DEVICES=0,1 \
mpirun -np 16 ./build-cuda/genesis4 examples/Example4-HGHG/Example4_a.cuda_profile.in
```

默认 `local_rank` 策略下，双 GPU、16 ranks 会形成：

```
rank 0 -> GPU0    rank 1 -> GPU1
rank 2 -> GPU0    rank 3 -> GPU1
...
```

查看 rank-to-device mapping：

```bash
GENESIS_CUDA_VERBOSE_DEVICE=1 \
GENESIS_CUDA_PRINT_DEVICE_SUMMARY=1 \
mpirun -np 16 ./build-cuda/genesis4 input.in
```

### 6.3 example 目录下提供的快捷脚本

```bash
# 单 GPU，8 ranks（默认）
./examples/Example4-HGHG/run_gpu.sh

# 双 GPU，16 ranks
NRANK=16 GPUS=0,1 ./examples/Example4-HGHG/run_gpu.sh

# Example3 同理
./examples/Example3-TimeDependent/run_gpu.sh
```

---

## 7. MPS 推荐运行

多 MPI ranks 共享同一张 GPU 时，建议测试 NVIDIA MPS。MPS 不由程序内部启动，需要用户或作业脚本显式启动。

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-pipe-$USER
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$USER
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

nvidia-cuda-mps-control -d

CUDA_VISIBLE_DEVICES=0,1 \
mpirun -np 16 \
  -x CUDA_MPS_PIPE_DIRECTORY \
  -x CUDA_MPS_LOG_DIRECTORY \
  ./build-cuda/genesis4 input.in

echo quit | nvidia-cuda-mps-control
```

example 目录下有快捷脚本 `run_gpu_mps.sh`，会自动启动 MPS daemon 并在退出时清理。

A/B 测试：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
tools/cuda_stage3_9C_mps_sweep.sh \
  ./build-cuda/genesis4 \
  examples/Example4-HGHG/Example4_a.cuda_profile.in \
 2,4,8,12
```

> MPS 是否有收益与 GPU、MPI rank 数、算例规模和集群环境有关。本版本验证：Example4 + 双 GPU + 8 ranks，MPS ON 比 OFF 大约快 20–25%。

---


### 7.1 Profiling

```bash
# Nsight Systems
tools/cuda_stage3_6_profile.sh \
  ./build-cuda/genesis4 \
  examples/Example4-HGHG/Example4_a.cuda_profile.in 16

# Nsight Compute
NCU_KERNEL_REGEX='beamLongitudinalOneFieldCachedInterpKernel' \
tools/cuda_stage3_6_ncu_profile.sh \
  ./build-cuda/genesis4 \
  examples/Example4-HGHG/Example4_a.cuda_profile.in 16
```

`ERR_NVGPUCTRPERM` 表示没有 GPU performance counter 权限，需要管理员调整。

---

## 8. 已验证性能摘要
（硬件配置 AMD EPYC 7H12 ,A100 80G）

大case，Example3-sample=12 ：

| 配置 | Wall Clock | 相对 CPU ADI 基线 |
|---|---:|---:|
| CPU ，100 核 | 6750 s | 1× | 
| CUDA FFT，4 核，MPS ON | 706 s | 9.6× |


小case，Example4-HGHG，小case：

| 配置 | Wall Clock | 相对 CPU ADI 基线 |
|---|---:|---:|
| CPU ，100 ranks | 140 s | 1× |
| CUDA FFT，4 ranks，MPS ON | 39 s | 3.6× |
| CUDA FFT，16 ranks，MPS ON | 35 s | 4× |

说明：
- HDF5 物理量差异在可忽略范围内（rel_max ≤ 1e-8）。

---

## 9. 已知限制与 FAQ

### 9.1 已知限制

1. CUDA FFT 是当前主优化路径；ADI CUDA path 不是当前主线。
2. MPS 是运行环境优化；部分集群可能限制用户启动 MPS。
3. CUDA-aware MPI 尚未作为默认路径实现；当前 multi-rank slippage 使用 boundary slice host staging。
4. `one4one` / sorting / CPU-only physics 需要单独验证。
5. 数值正确性是容差一致，（对比时请注意genesis1.3的编译方式，FFTW or ADI），ADI solver 与FFT solver 本身存在小偏差。
6. CPU ADI vs CUDA FFT 加速比不是严格 solver-to-solver benchmark。

### 11 FAQ

**Q：程序没有使用 CUDA kernel？**
检查输入文件是否包含：

```text
cuda_fieldsolver = true
fft_fieldsolver = true
```

**Q：多 GPU 只用 GPU0？**
检查：

```bash
GENESIS_CUDA_VERBOSE_DEVICE=1
GENESIS_CUDA_PRINT_DEVICE_SUMMARY=1
```

是否设置了：

```bash
GENESIS_CUDA_DEVICE=0
CUDA_VISIBLE_DEVICES=0
```

**Q：数值回归时想关闭所有 fast path？**
```bash
GENESIS_CUDA_SAFE_MODE=1
```

需要更保守可叠加 `GENESIS_CUDA_BIND_FFT_FIELD=0` 等。

**Q：MPS 启动失败？**
常见原因：

- 没有 `nvidia-cuda-mps-control`；
- 没有权限；
- 旧的 MPS daemon 没退出；
- `CUDA_MPS_PIPE_DIRECTORY` 或 `CUDA_MPS_LOG_DIRECTORY` 不存在。

先 `echo quit | nvidia-cuda-mps-control`，再重新启动。

**Q：是否必须使用 MPS？**
不是。MPS 是推荐生产运行方式，不是功能依赖。无 MPS 也能运行，只是多 rank/GPU 下可能性能较低。

**Q：是否可以用更多 MPI ranks？**
可以，但不一定更快。建议先 sweep ranks/GPU。MPS ON 下 4–16 ranks/GPU 通常都稳定；无 MPS 时 8 ranks/GPU 是较好起点。

---

## 致谢

本仓库基于 [Sven Reiche 的 Genesis-1.3-Version4](https://github.com/svenreiche/Genesis-1.3-Version4)。CUDA 改造与运行配套工具是本仓库的增量贡献，其余代码版权与许可证遵循上游仓库。
