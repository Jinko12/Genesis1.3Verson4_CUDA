#!/usr/bin/env bash
set -euo pipefail

ARCHES="${GENESIS_CUDA_ARCHITECTURES:-80}"
CPU_BUILD="${CPU_BUILD:-build-cpu}"
CUDA_BUILD="${CUDA_BUILD:-build-cuda}"

cmake -S . -B "$CPU_BUILD" -DUSE_CUDA=OFF
cmake --build "$CPU_BUILD" -j "${JOBS:-$(nproc)}"

cmake -S . -B "$CUDA_BUILD" -DUSE_CUDA=ON -DGENESIS_CUDA_ARCHITECTURES="$ARCHES"
cmake --build "$CUDA_BUILD" -j "${JOBS:-$(nproc)}"
