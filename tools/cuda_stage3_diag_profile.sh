#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 /path/to/genesis4 input.in [extra genesis args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GENESIS_BIN="$1"
INPUT_FILE="$2"
shift 2

ROOT="${ROOT:-cuda-stage3-diag-profile-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$ROOT/diag-on" "$ROOT/diag-off"

run_profile() {
  local name="$1"
  local diag="$2"
  local dir="$ROOT/$name"
  OUTDIR="$dir" \
  GENESIS_CUDA_DEFER_FIELD_D2H=1 \
  GENESIS_CUDA_BIND_FFT_FIELD=1 \
  GENESIS_CUDA_DIAG_REDUCTION="$diag" \
  "$SCRIPT_DIR/cuda_profile.sh" "$GENESIS_BIN" "$INPUT_FILE" "$@"
}

run_profile diag-on 1 "$@"
run_profile diag-off 0 "$@"

echo "Stage-3 diagnostic profile outputs are in $ROOT"
echo "Expected: diag-on should avoid full beam/field D2H during standard diagnostics; diag-off is the conservative comparison."
