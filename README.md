# Genesis 1.3 CUDA GPU Resident

[English](./README.md) | [中文](./README_CN.md)

**Version:** `GPUResident Stage 4.2B`
**Upstream baseline:** [Genesis-1.3-Version4](https://github.com/svenreiche/Genesis-1.3-Version4) (4.6.12)

This repository adds a CUDA GPU acceleration path on top of Sven Reiche's Genesis 1.3 Version 4. The goal is to keep **beam, field, source deposition, field solve, diagnostics, and slippage as GPU resident as much as possible**, and to synchronize with the host only for HDF5 output, CPU fallback, required MPI boundaries, or explicit debugging.

**A one-step switch `GENESIS_CUDA_SAFE_MODE=1` can be used to disable all optional fast paths.**

> Upstream copyright and licensing follow Sven Reiche's original repository. This repository only modifies and supplements the CUDA path and runtime support tools. Maintained regularly.


---

## Table of Contents

- [1. Version Positioning and Core Principles](#1-version-positioning-and-core-principles)
- [2. Recommended Hardware and Software Environment](#2-recommended-hardware-and-software-environment)
- [3. Build](#3-build)
- [4. Input File Configuration](#4-input-file-configuration)
- [5. Default Built-in CUDA Behavior](#5-default-built-in-cuda-behavior)
- [6. Basic Run Modes](#6-basic-run-modes)
- [7. Recommended MPS Run Mode](#7-recommended-mps-run-mode)
- [8. Verified Performance Summary](#8-verified-performance-summary)
- [9. Known Limitations and FAQ](#9-known-limitations-and-faq)

---

## 1. Version Positioning and Core Principles

- Add a GPU-resident main path while preserving the original Genesis physics model and input-file semantics.
- CUDA optimization paths that have been verified as stable are enabled by default.
- Keep all CPU fallback paths and safe-mode switches available.
- **The solver choice is not changed automatically for the user**: whether to enable the CUDA FFT field solver is still determined by `cuda_fieldsolver=true` / `fft_fieldsolver=true` in the input file.
- **MPS is not started automatically**: MPS is a runtime-environment policy controlled by the user or job script.

---

## 2. Recommended Hardware and Software Environment

### 2.1 Hardware

- NVIDIA GPU: A100, H100, L40S, RTX 6000 Ada, RTX 4090, or similar GPUs are recommended.
- GPU memory: ≥ 24 GB is recommended; large-scale slices/particles/multiple fields require more.
- CPU: ≥ 16 cores.
- System memory: ≥ 64 GB.

### 2.2 Software

- Linux x86_64
- CMake 3.18+
- GCC/G++ with C++17 support
- CUDA Toolkit (NVCC, cuFFT, cuDART)
- MPI: OpenMPI or MPICH
- HDF5: parallel HDF5 is recommended
- FFTW (optional, for CPU FFT fallback or CPU comparison)

---

## 3. Build

```bash
cmake -S . -B build-cuda \
  -DUSE_CUDA=ON \
  -DGENESIS_CUDA_ARCHITECTURES=80

cmake --build build-cuda -j
```

Adjust `GENESIS_CUDA_ARCHITECTURES` according to the GPU architecture:

| GPU | Recommended value |
|---|---:|
| V100 | 70 |
| RTX 20 series / T4 | 75 |
| A100 / A30 | 80 |
| RTX 30 series | 86 |
| L40S / RTX 6000 Ada / RTX 40 series | 89 |
| H100 | 90 |

If HDF5 is in a conda environment:

```bash
cmake -S . -B build-cuda \
  -DUSE_CUDA=ON \
  -DGENESIS_CUDA_ARCHITECTURES=80 \
  -DHDF5_ROOT=/path/to/conda/env
```

Build output: `build-cuda/genesis4`

### CPU Reference Build (optional)

```bash
cmake -S . -B build-cpu -DUSE_CUDA=OFF
cmake --build build-cpu -j
```

> For strict performance comparisons, CPU ADI, CPU FFTW, and CUDA FFT should be distinguished. It is not recommended to treat CPU ADI vs CUDA FFT as a strict solver-to-solver comparison.

---

## 4. Input File Configuration

Enable the CUDA FFT field solver in `&track`:

```text
&track
  fft_fieldsolver = true
  cuda_fieldsolver = true
&end
```

`cuda_beam` follows `cuda_fieldsolver` by default, so it usually does not need to be set separately.

Example inputs provided in this repository:

```
examples/Example3-TimeDependent/Example3.cuda.in
examples/Example4-HGHG/Example4_a.cuda.in
examples/Example4-HGHG/Example4_a.cuda_profile.in   # for profiling; lattice paths are relative to the repository root
```

---

## 5. Default Built-in CUDA Behavior

The following optimization defaults have already been built into this version, so users **no longer need to set them manually**:

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

Meaning overview:

| Default item | Default value | Purpose |
|---|---:|---|
| `GENESIS_CUDA_DEVICE_POLICY` | `local_rank` | Round-robin mapping from MPI local ranks to visible GPUs |
| `GENESIS_CUDA_FAST_KERNELS` | 1 | Enable verified CUDA fast kernels |
| `GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD` | 1 | Cache field interpolation within longitudinal RK steps |
| `GENESIS_CUDA_INPLACE_SLIPPAGE` | 1 | Use in-place CUDA slippage to avoid scratch D2D copies |
| `GENESIS_CUDA_LONGITUDINAL_ALGEBRA_OPT` | 1 | FP64 algebraically equivalent simplification |
| `GENESIS_CUDA_BIND_FFT_FIELD` | 1 | Bind BeamSolver directly to the FFT field device buffer |
| `GENESIS_CUDA_DEFER_FIELD_D2H` | 1 | Defer field D2H by default and synchronize only at required boundaries |
| `GENESIS_CUDA_MPI_SLIPPAGE` | 1 | Use the CUDA-resident path by default for multi-rank slippage |
| `GENESIS_CUDA_DIAG_REDUCTION` | 1 | Use GPU compact reduction by default for diagnostics |

Each variable can still be explicitly set to 0 to force fallback.

---

## 6. Basic Run Modes

### 6.1 Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 \
mpirun -np 8 ./build-cuda/genesis4 examples/Example4-HGHG/Example4_a.cuda_profile.in
```

In practice, using more ranks on a single GPU is not always faster. It is recommended to first sweep `2, 4, 8, 12 ranks/GPU`:

```bash
CUDA_VISIBLE_DEVICES=0 \
tools/cuda_stage3_9_worker_sweep.sh \
  ./build-cuda/genesis4 \
  examples/Example4-HGHG/Example4_a.cuda_profile.in \
  1,2,4,8,12,16,24,32
```

### 6.2 Multi-GPU

```bash
CUDA_VISIBLE_DEVICES=0,1 \
mpirun -np 16 ./build-cuda/genesis4 examples/Example4-HGHG/Example4_a.cuda_profile.in
```

With the default `local_rank` policy, two GPUs with 16 ranks will form the following mapping:

```
rank 0 -> GPU0    rank 1 -> GPU1
rank 2 -> GPU0    rank 3 -> GPU1
...
```

To check the rank-to-device mapping:

```bash
GENESIS_CUDA_VERBOSE_DEVICE=1 \
GENESIS_CUDA_PRINT_DEVICE_SUMMARY=1 \
mpirun -np 16 ./build-cuda/genesis4 input.in
```

### 6.3 Convenience scripts provided under the example directories

```bash
# Single GPU, 8 ranks by default
./examples/Example4-HGHG/run_gpu.sh

# Two GPUs, 16 ranks
NRANK=16 GPUS=0,1 ./examples/Example4-HGHG/run_gpu.sh

# Same for Example3
./examples/Example3-TimeDependent/run_gpu.sh
```

---

## 7. Recommended MPS Run Mode

When multiple MPI ranks share the same GPU, it is recommended to test NVIDIA MPS. MPS is not started internally by the program; it must be started explicitly by the user or job script.

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

A convenience script `run_gpu_mps.sh` is provided under the example directories. It automatically starts the MPS daemon and cleans it up on exit.

A/B testing:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
tools/cuda_stage3_9C_mps_sweep.sh \
  ./build-cuda/genesis4 \
  examples/Example4-HGHG/Example4_a.cuda_profile.in \
 2,4,8,12
```

> Whether MPS provides a benefit depends on the GPU, number of MPI ranks, case size, and cluster environment.

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

`ERR_NVGPUCTRPERM` means that the user does not have GPU performance-counter permissions and that administrator adjustment is required.

---

## 8. Verified Performance Summary
(Hardware configuration: AMD EPYC 7H12, A100 80G)

Large case, Example3-sample=12:

| Configuration | Wall Clock | Relative to CPU ADI baseline |
|---|---:|---:|
| CPU, 100 cores | 6750 s | 1× | 
| CUDA FFT, 4 cores, MPS ON | 706 s | 9.6× |


Small case, Example4-HGHG:

| Configuration | Wall Clock | Relative to CPU ADI baseline |
|---|---:|---:|
| CPU, 100 ranks | 140 s | 1× |
| CUDA FFT, 4 ranks, MPS ON | 39 s | 3.6× |
| CUDA FFT, 16 ranks, MPS ON | 35 s | 4× |

Notes:
- Differences in HDF5 physical quantities are within a negligible range (rel_max ≤ 1e-8).

---

## 9. Known Limitations and FAQ

### 9.1 Known Limitations

1. CUDA FFT is the current main optimization path; the ADI CUDA path is not the current focus.
2. MPS is a runtime-environment optimization; some clusters may restrict users from starting MPS.
3. CUDA-aware MPI has not yet been implemented as the default path; the current multi-rank slippage uses boundary-slice host staging.
4. `one4one` / sorting / CPU-only physics, such as space charge, need to be validated separately.
5. Numerical correctness is tolerance-based. When comparing results, please pay attention to how Genesis 1.3 is built, such as FFTW or ADI. The ADI solver and FFT solver have small inherent differences.
6. The CPU ADI vs CUDA FFT speedup is not a strict solver-to-solver benchmark.
7. Large cases such as Example3 require substantial GPU memory. OOM (out of memory/money🐶) on machines with small GPU memory is unavoidable. This version already inherits the excellent memory planning of the original Genesis, and future work may target more compact GPU-memory usage.

### 11 FAQ

**Q: The program does not use CUDA kernels.**
Check whether the input file contains:

```text
cuda_fieldsolver = true
fft_fieldsolver = true
```

**Q: Multi-GPU only uses GPU0.**
Check:

```bash
GENESIS_CUDA_VERBOSE_DEVICE=1
GENESIS_CUDA_PRINT_DEVICE_SUMMARY=1
```

Check whether the following variables are set:

```bash
GENESIS_CUDA_DEVICE=0
CUDA_VISIBLE_DEVICES=0
```

**Q: I want to disable all fast paths for numerical regression.**
```bash
GENESIS_CUDA_SAFE_MODE=1
```

For a more conservative run, additional variables such as `GENESIS_CUDA_BIND_FFT_FIELD=0` can be combined.

**Q: MPS fails to start.**
Common reasons:

- `nvidia-cuda-mps-control` is not available;
- insufficient permission;
- an old MPS daemon has not exited;
- `CUDA_MPS_PIPE_DIRECTORY` or `CUDA_MPS_LOG_DIRECTORY` does not exist.

First run `echo quit | nvidia-cuda-mps-control`, then start it again.

**Q: Is MPS required?**
No. MPS is a recommended production run mode, not a functional dependency. The program can run without MPS, but performance may be lower when multiple ranks share one GPU.

**Q: Can I use more MPI ranks?**
Yes, but it may not be faster. It is recommended to sweep ranks/GPU first. With MPS ON, 4–16 ranks/GPU are usually stable; without MPS, 8 ranks/GPU is a good starting point.

---

## Acknowledgements

This repository is based on [Sven Reiche's Genesis-1.3-Version4](https://github.com/svenreiche/Genesis-1.3-Version4). The CUDA modifications and runtime support tools are incremental contributions from this repository. Copyright and licensing for the remaining code follow the upstream repository.
