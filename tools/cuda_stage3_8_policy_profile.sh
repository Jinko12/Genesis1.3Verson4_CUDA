#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<'USAGE'
Usage: tools/cuda_stage3_8_policy_profile.sh /path/to/genesis4 input.in np [output_dir]

Compares three policies for the same MPI rank count:
  forced0      all ranks forced to visible device 0
  round_robin  local_rank % visible_device_count
  block        contiguous local-rank blocks per GPU
USAGE
  exit 2
fi

GENESIS_BIN=$1
INPUT=$2
NP=$3
OUTDIR=${4:-stage3_8_policy_profile_np${NP}_$(date +%Y%m%d_%H%M%S)}
export GENESIS_CUDA_STAGE38_MODES=forced0,round_robin,block
"$(dirname "$0")/cuda_stage3_8_rank_sweep.sh" "$GENESIS_BIN" "$INPUT" "$NP" "$OUTDIR"
