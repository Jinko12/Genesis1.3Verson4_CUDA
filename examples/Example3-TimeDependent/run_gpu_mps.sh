#!/usr/bin/env bash
# Example3 GPU 运行示例（MPS ON，多 ranks/GPU 共享）。
#
# Simplified 版已经把推荐默认值内置，本脚本只负责启动 MPS daemon
# 与传递 MPS pipe/log 目录。
#
# 用法：
#   ./run_gpu_mps.sh                      # 默认 16 ranks, 双 GPU
#   NRANK=8 GPUS=0 ./run_gpu_mps.sh       # 单 GPU 8 ranks
#
# 集群不允许用户启 MPS 时，请改用 run_gpu.sh。

set -euo pipefail
cd "$(dirname "$0")"

NRANK="${NRANK:-16}"
GPUS="${GPUS:-0,1}"
GENESIS="${GENESIS:-../../build-cuda/genesis4}"
INPUT="${INPUT:-Example3.cuda.in}"

if [[ ! -x "$GENESIS" ]]; then
  echo "ERROR: genesis4 not found at $GENESIS"
  exit 1
fi

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-pipe-$USER"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-$USER"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

if ! pgrep -u "$USER" -f nvidia-cuda-mps-control >/dev/null; then
  echo "Starting MPS daemon..."
  nvidia-cuda-mps-control -d
  sleep 1
fi

cleanup_mps() {
  echo "Stopping MPS daemon..."
  echo quit | nvidia-cuda-mps-control || true
}
trap cleanup_mps EXIT

echo "=== Example3 GPU run with MPS ==="
echo "GENESIS = $GENESIS"
echo "INPUT   = $INPUT"
echo "NRANK   = $NRANK"
echo "GPUS    = $GPUS"
echo "MPS pipe= $CUDA_MPS_PIPE_DIRECTORY"
echo ""

CUDA_VISIBLE_DEVICES="$GPUS" \
mpirun -np "$NRANK" \
  -x CUDA_MPS_PIPE_DIRECTORY \
  -x CUDA_MPS_LOG_DIRECTORY \
  "$GENESIS" "$INPUT"
