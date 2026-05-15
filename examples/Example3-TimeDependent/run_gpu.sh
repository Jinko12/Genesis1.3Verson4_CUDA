#!/usr/bin/env bash
# Example3 GPU 运行示例（无 MPS）。
#
# Simplified 版已经把推荐默认值内置到 genesis4，正常运行不需要
# 任何 GENESIS_CUDA_* 环境变量。CUDA solver 是否启用，由输入文件
# Example3.cuda.in 的 &track 中的 cuda_fieldsolver/fft_fieldsolver 控制。
#
# 用法：
#   ./run_gpu.sh                      # 默认 8 ranks，GPU 0
#   NRANK=16 GPUS=0,1 ./run_gpu.sh    # 16 ranks，双 GPU round-robin
#   GENESIS=/abs/path/genesis4 ./run_gpu.sh

set -euo pipefail
cd "$(dirname "$0")"

NRANK="${NRANK:-8}"
GPUS="${GPUS:-0}"
GENESIS="${GENESIS:-../../build-cuda/genesis4}"
INPUT="${INPUT:-Example3.cuda.in}"

if [[ ! -x "$GENESIS" ]]; then
  echo "ERROR: genesis4 not found at $GENESIS"
  echo "Build first with:"
  echo "  cmake -S . -B build-cuda -DUSE_CUDA=ON -DGENESIS_CUDA_ARCHITECTURES=80"
  echo "  cmake --build build-cuda -j"
  exit 1
fi

echo "=== Example3 GPU run ==="
echo "GENESIS = $GENESIS"
echo "INPUT   = $INPUT"
echo "NRANK   = $NRANK"
echo "GPUS    = $GPUS"
echo ""

CUDA_VISIBLE_DEVICES="$GPUS" \
mpirun -np "$NRANK" "$GENESIS" "$INPUT"
