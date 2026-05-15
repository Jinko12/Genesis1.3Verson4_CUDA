#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 /path/to/cpu/genesis4 /path/to/cuda/genesis4 input.in [extra genesis args...]" >&2
  exit 2
fi

CPU_BIN="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
CUDA_BIN="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
INPUT_FILE="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
shift 3

ROOT="${ROOT:-cuda-regression-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$ROOT/cpu" "$ROOT/cuda-eager" "$ROOT/cuda-resident" "$ROOT/cuda-resident-bind-off" "$ROOT/cuda-resident-diag-off"

run_case() {
  local name="$1"
  local bin="$2"
  shift 2
  local dir="$ROOT/$name"
  ( cd "$dir" && "$bin" "$@" "$INPUT_FILE" ) > "$dir/stdout.log" 2> "$dir/stderr.log"
}

run_case cpu "$CPU_BIN" "$@"

GENESIS_CUDA_DEFER_FIELD_D2H=0 GENESIS_CUDA_BIND_FFT_FIELD=0 GENESIS_CUDA_DIAG_REDUCTION=0 \
  run_case cuda-eager "$CUDA_BIN" "$@"

GENESIS_CUDA_DEFER_FIELD_D2H=1 GENESIS_CUDA_BIND_FFT_FIELD=1 GENESIS_CUDA_DIAG_REDUCTION=1 \
  run_case cuda-resident "$CUDA_BIN" "$@"

GENESIS_CUDA_DEFER_FIELD_D2H=1 GENESIS_CUDA_BIND_FFT_FIELD=0 GENESIS_CUDA_DIAG_REDUCTION=1 \
  run_case cuda-resident-bind-off "$CUDA_BIN" "$@"

GENESIS_CUDA_DEFER_FIELD_D2H=1 GENESIS_CUDA_BIND_FFT_FIELD=1 GENESIS_CUDA_DIAG_REDUCTION=0 \
  run_case cuda-resident-diag-off "$CUDA_BIN" "$@"

echo "Regression outputs are in $ROOT"
echo "Next: compare key HDF5 datasets with h5diff or a domain-specific Python comparator."
