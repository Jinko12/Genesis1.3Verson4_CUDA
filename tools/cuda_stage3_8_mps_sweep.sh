#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<'USAGE'
Usage: tools/cuda_stage3_8_mps_sweep.sh /path/to/genesis4 input.in ranks_csv [output_dir]

Runs a strict single-GPU rank sweep with CUDA MPS off and on.
Example:
  CUDA_VISIBLE_DEVICES=0 tools/cuda_stage3_8_mps_sweep.sh ./build-cuda/genesis4 examples/Example4-HGHG/Example4_a.cuda_profile.in 4,8,16,24,32
USAGE
  exit 2
fi

GENESIS_BIN=$1
INPUT=$2
RANKS_CSV=$3
OUTDIR=${4:-stage3_8_mps_sweep_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTDIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export GENESIS_CUDA_STAGE38_MODES=single

echo "[stage3.8] MPS off sweep"
"$(dirname "$0")/cuda_stage3_8_rank_sweep.sh" "$GENESIS_BIN" "$INPUT" "$RANKS_CSV" "$OUTDIR/mps_off"

if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps-$USER}
  export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log-$USER}
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d
  trap 'echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true' EXIT
  echo "[stage3.8] MPS on sweep"
  "$(dirname "$0")/cuda_stage3_8_rank_sweep.sh" "$GENESIS_BIN" "$INPUT" "$RANKS_CSV" "$OUTDIR/mps_on"
  echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
  trap - EXIT
else
  echo "nvidia-cuda-mps-control not found; skipped MPS-on sweep" | tee "$OUTDIR/mps_skipped.txt"
fi

find "$OUTDIR" -name summary.tsv -print -exec cat {} \;
