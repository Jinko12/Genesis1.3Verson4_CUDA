# Genesis 1.3 CUDA GPU Resident（简化版）

**版本：** `GPUResident Stage 4.2B Simplified (Final, compile-fix)`
**上游基线：** [Genesis-1.3-Version4](https://github.com/svenreiche/Genesis-1.3-Version4) (4.6.12 / 4.6.x 分支)

本仓库在 Sven Reiche 的 Genesis 1.3 Version 4 之上加入 CUDA GPU 加速路径，目标是让 **beam、field、source deposition、field solve、diagnostics 和 slippage 尽可能保持 GPU resident**，只在 HDF5 输出、CPU fallback、必要 MPI 边界或显式调试时才同步到 host。

**Simplified 版的核心改动：把已经验证稳定的优化路径全部 *内置为默认开启*。** 用户正常运行不再需要写一长串环境变量；只要输入文件启用 CUDA solver，就能拿到完整加速。如果需要数值回归或调试，可用一键开关 `GENESIS_CUDA_SAFE_MODE=1` 关闭所有可选 fast path。

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
- [8. 安全模式与调试回退](#8-安全模式与调试回退)
- [9. 主要修改内容](#9-主要修改内容)
- [10. 验证与回归](#10-验证与回归)
- [11. 已验证性能摘要](#11-已验证性能摘要)
- [12. 已知限制与 FAQ](#12-已知限制与-faq)

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
  zstop = 1.14
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

Simplified 版已经把下列优化的默认值内置到程序中，用户**不再需要手动设置**：

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

每个变量都仍可显式设为 0 强制回退（见 [§8 安全模式与调试回退](#8-安全模式与调试回退)）。

---

## 6. 基本运行方式

### 6.1 单 GPU

```bash
CUDA_VISIBLE_DEVICES=0 \
mpirun -np 8 ./build-cuda/genesis4 examples/Example4-HGHG/Example4_a.cuda_profile.in
```

经验上单 GPU 不是 rank 越多越快，建议先 sweep `4, 8, 12, 16 ranks/GPU`：

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
  4,8,12,16
```

> MPS 是否有收益与 GPU、MPI rank 数、算例规模和集群环境有关。本版本验证：Example4 + 双 GPU + 16 ranks，MPS ON 比 OFF 大约快 20–25%。

---

## 8. 安全模式与调试回退

### 8.1 安全模式

```bash
GENESIS_CUDA_SAFE_MODE=1 \
mpirun -np 16 ./build-cuda/genesis4 input.in
```

`SAFE_MODE` 会关闭以下可选 fast path：

```
GENESIS_CUDA_FAST_KERNELS
GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD
GENESIS_CUDA_INPLACE_SLIPPAGE
GENESIS_CUDA_LONGITUDINAL_ALGEBRA_OPT
GENESIS_CUDA_SYMMETRIC_TRANSVERSE
```

**不会**关闭 CUDA solver 本身，也不会改变输入文件中选择的 solver。

### 8.2 完全保守 CUDA 调试

如果还需要更保守，可以同时把所有 GPU resident 路径回退到旧的 D2H/H2D 行为：

```bash
GENESIS_CUDA_SAFE_MODE=1 \
GENESIS_CUDA_BIND_FFT_FIELD=0 \
GENESIS_CUDA_DEFER_FIELD_D2H=0 \
GENESIS_CUDA_MPI_SLIPPAGE=0 \
GENESIS_CUDA_DIAG_REDUCTION=0 \
mpirun -np 16 ./build-cuda/genesis4 input.in
```

### 8.3 调试 / 观测开关

| 环境变量 | 用途 |
|---|---|
| `GENESIS_CUDA_VERBOSE_DEVICE=1` | 打印每个 rank 的 GPU 绑定 |
| `GENESIS_CUDA_PRINT_DEVICE_SUMMARY=1` | 打印每 GPU 的 rank 数汇总 |
| `GENESIS_CUDA_NVTX=1` | 开启 NVTX range 标注 |
| `GENESIS_CUDA_MEMORY_AUDIT=1` | 输出 CUDA allocation / cuFFT workspace 审计 |

---

## 9. 主要修改内容

### 9.1 Beam SoA 与 GPU beam tracking

将 beam 粒子状态从 CPU AoS/逐 slice 结构转换为 GPU device SoA：

```
gamma, theta, x/y, px/py, ez, particle_slice, slice_start/slice_count
```

- 粒子推进使用 CUDA kernel；
- longitudinal RK4 在 GPU 上完成；
- transverse tracking 用专门 kernel；
- 只在 CPU-only physics 或 host 输出时同步。

### 9.2 FFT field solver GPU resident

新增 CUDA FFT field solver，用 cuFFT batched transform 处理多 slice field propagation。

```
build source on GPU
batched cuFFT forward
propagation / filter multiply
batched cuFFT inverse
normalization
```

cuFFT plan 在 solver 生命周期内复用。

### 9.3 BeamSolver 直接绑定 FFT field device buffer

旧路径：

```
field GPU -> host Field::field -> beam GPU
```

新路径：

```
FieldSolverFFTCuda device field
  -> BeamSolver device pointer table
  -> beam longitudinal kernel 直接读取
```

回退：`GENESIS_CUDA_BIND_FFT_FIELD=0`

### 9.4 Source deposition 直接读取 Beam SoA

```
GPU Beam SoA -> source deposition kernel -> GPU source grid -> cuFFT propagation
```

只有 Beam CUDA state 不可用时才回退 host `Particle*` staging。

### 9.5 Diagnostics GPU compact reduction

diagnostics 不再默认下载完整 beam / field，host 端只下载 compact 结果：

```
beam:  per-slice moments, bunching, energy / spread
field: power, phase, near/far-field compact moments
```

历史验证：diagnostics/slippage resident 修复后，D2H 从 full-sync 路径的 125,583.84 MB 降至 9.36 MB。

### 9.6 CUDA-resident slippage 与 in-place slippage

- **单 rank**：field slippage 直接在 CUDA FFT field buffer 上完成。
- **多 rank**：只交换 boundary slice，不下载完整 field record。
- **Stage 4.0A in-place**：消除 scratch D2D。

```
field -> scratch -> shifted field   (旧)
field -> field                       (新, in-place kernel)
```

验证：

```
D2D total: 31,212 MB -> 0 MB
D2D calls: 120 -> 0
HDF5 correctness: 81/81 datasets passed
```

### 9.7 multi-rank CUDA slippage boundary

```
outgoing boundary slice D2H
MPI exchange one slice
incoming boundary slice H2D
CUDA in-place shift / boundary injection
```

### 9.8 rank-to-device mapping 与 worker budget

```bash
GENESIS_CUDA_DEVICE_POLICY=local_rank   # 默认
# 可选：single, block, world_rank
GENESIS_CUDA_DEVICE=0                   # 强制全部 rank 用 GPU0
GENESIS_CUDA_MAX_RANKS_PER_DEVICE=8
GENESIS_CUDA_STRICT_RANK_BUDGET=1
```

### 9.9 Longitudinal RK FP64 代数等价优化（Stage 4.2B）

NCU 诊断 `beamLongitudinalOneFieldCachedInterpKernel` 为 FP64 math-bound（Compute ~88%，FP64 pipeline ~88%，DRAM 仅 ~5%，occupancy ~99%）。Stage 4.2B 做了保守整理：

```
invGam = 1 / gamma
invBtpar0 = 1 / sqrt(1 - btper0 * invGam * invGam)
```

不使用 fast-math、不使用 mixed precision、不降低 RK 阶数。

验证结果：

```
Field/power rel_max:       4.2e-12
Beam/energy rel_max:       3.0e-15
Beam/energyspread rel_max: 1.2e-8
Beam/bunching rel_max:     3.4e-12
```

性能：MPS ON 下额外 +5.4%。

### 9.10 安全模式（Simplified 版新增）

```bash
GENESIS_CUDA_SAFE_MODE=1
```

一键关闭所有可选 fast path，便于数值回归与新算例排查。

---

## 10. 验证与回归

### 10.1 HDF5 correctness guard

```bash
python3 tools/cuda_stage3_9B_correctness_guard.py \
  cpu_or_reference/Example4_a.out.h5 \
  cuda/Example4_a.out.h5 \
  --rtol 1e-8 --atol 1e-10
```

> `Meta/TimeStamp`、`Meta/cwd` 等运行元数据可能不同，这不是物理差异。

### 10.2 性能测试

Rank/GPU sweep：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
tools/cuda_stage3_9_worker_sweep.sh \
  ./build-cuda/genesis4 \
  examples/Example4-HGHG/Example4_a.cuda_profile.in \
  1,2,4,8,12,16
```

MPS sweep：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
tools/cuda_stage3_9C_mps_sweep.sh \
  ./build-cuda/genesis4 \
  examples/Example4-HGHG/Example4_a.cuda_profile.in \
  4,8,12,16
```

In-place slippage / algebra A/B：

```bash
tools/cuda_stage4_0A_inplace_slippage_profile.sh \
  ./build-cuda/genesis4 \
  examples/Example4-HGHG/Example4_a.cuda_profile.in \
  16

tools/cuda_stage4_2B_longitudinal_algebra_profile.sh \
  ./build-cuda/genesis4 \
  examples/Example4-HGHG/Example4_a.cuda_profile.in \
  16
```

### 10.3 Profiling

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

## 11. 已验证性能摘要

Example4-HGHG 全程：

| 配置 | Wall Clock | 相对 CPU ADI 基线 |
|---|---:|---:|
| CPU ADI，32 核 | 1647 s | 1× |
| CPU FFTW，42 核（Genesis 4.6.12） | 299 s | 5.5× |
| CUDA FFT，16 核，MPS OFF | 72.4 s | 22.7× |
| CUDA FFT，16 核，MPS ON | 59.3 s | 27.8× |
| CUDA FFT + ALGEBRA_OPT，16 核，MPS ON | 56.1 s | 29.4× |

说明：

- CPU 基线为 ADI 路径，CUDA 为 FFT 路径。
- 若用于正式论文或报告，应补充 CPU FFTW vs CUDA FFT 的严格 solver-to-solver 对照。
- HDF5 物理量差异在可忽略范围内（rel_max ≤ 1e-8）。

---

## 12. 已知限制与 FAQ

### 12.1 已知限制

1. CUDA FFT 是当前主优化路径；ADI CUDA path 不是当前主线。
2. MPS 是运行环境优化；部分集群可能限制用户启动 MPS。
3. CUDA-aware MPI 尚未作为默认路径实现；当前 multi-rank slippage 使用 boundary slice host staging。
4. `one4one` / sorting / CPU-only physics 需要单独验证。
5. 数值正确性是容差一致，不保证 bitwise identical。
6. CPU ADI vs CUDA FFT 加速比不是严格 solver-to-solver benchmark。

### 12.2 FAQ

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

需要更保守可叠加 `GENESIS_CUDA_BIND_FFT_FIELD=0` 等（见 §8.2）。

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
