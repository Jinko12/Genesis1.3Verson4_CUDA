#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 /path/to/genesis4 input.in [extra genesis args...]" >&2
  exit 2
fi

GENESIS_BIN="$1"
INPUT_FILE="$2"
shift 2

OUTDIR="${OUTDIR:-cuda-profile-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTDIR"

export GENESIS_CUDA_DEFER_FIELD_D2H="${GENESIS_CUDA_DEFER_FIELD_D2H:-1}"
export GENESIS_CUDA_BIND_FFT_FIELD="${GENESIS_CUDA_BIND_FFT_FIELD:-1}"
export GENESIS_CUDA_DIAG_REDUCTION="${GENESIS_CUDA_DIAG_REDUCTION:-1}"

nsys profile \
  --trace=cuda,osrt,mpi,nvtx \
  --cuda-memory-usage=true \
  --stats=true \
  --output="$OUTDIR/nsys" \
  "$GENESIS_BIN" "$@" "$INPUT_FILE"

# Optional focused kernel profile.  Set NC_KERNEL to a regex such as
# "beamLongitudinalKernel|buildSourceFromSoAKernel|fftPropagateKernel|beamMomentsDiagnosticKernel|fftFieldDiagnosticKernel|fftFieldFarfieldDiagnosticKernel".
if [[ -n "${NC_KERNEL:-}" ]]; then
  ncu --target-processes all \
      --kernel-name "$NC_KERNEL" \
      --set full \
      --export "$OUTDIR/ncu" \
      "$GENESIS_BIN" "$@" "$INPUT_FILE"
fi
