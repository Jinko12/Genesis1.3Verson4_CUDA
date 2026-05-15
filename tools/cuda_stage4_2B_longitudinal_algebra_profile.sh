#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 /path/to/genesis4 input.in [np]" >&2
  exit 2
fi

GENESIS_BIN="$1"
INPUT_FILE="$2"
NP="${3:-16}"
OUTDIR="${OUTDIR:-stage4_2B_longitudinal_algebra_$(date +%Y%m%d_%H%M%S)}"
MPIEXEC="${MPIEXEC:-mpirun}"
mkdir -p "$OUTDIR"

run_case() {
  local label="$1"
  local opt="$2"
  local dir="$OUTDIR/$label"
  mkdir -p "$dir"
  echo "[stage4_2B] running $label: GENESIS_CUDA_LONGITUDINAL_ALGEBRA_OPT=$opt np=$NP"
  (
    cd "$dir"
    GENESIS_CUDA_LONGITUDINAL_ALGEBRA_OPT="$opt" \
    GENESIS_CUDA_FAST_KERNELS="${GENESIS_CUDA_FAST_KERNELS:-1}" \
    GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD="${GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD:-1}" \
    GENESIS_CUDA_INPLACE_SLIPPAGE="${GENESIS_CUDA_INPLACE_SLIPPAGE:-1}" \
    GENESIS_CUDA_DEVICE_POLICY="${GENESIS_CUDA_DEVICE_POLICY:-local_rank}" \
    /usr/bin/time -p "$MPIEXEC" -np "$NP" "$GENESIS_BIN" "$INPUT_FILE" \
      > run.log 2>&1
  )
}

run_case algebra_on 1
run_case algebra_off 0

python3 - <<'PY' "$OUTDIR"
import pathlib, re, sys
root = pathlib.Path(sys.argv[1])
print("case\twall_clock_s\treal_s\texit")
for case in ["algebra_on", "algebra_off"]:
    log = root / case / "run.log"
    txt = log.read_text(errors="replace") if log.exists() else ""
    wall = ""
    real = ""
    exit_code = "0"
    m = re.search(r"Total Wall Clock Time:\s*([0-9.]+)", txt)
    if m: wall = m.group(1)
    m = re.search(r"^real\s+([0-9.]+)", txt, re.M)
    if m: real = m.group(1)
    if "Exit code" in txt:
        m = re.search(r"Exit code\s*[:=]\s*(\d+)", txt)
        if m: exit_code = m.group(1)
    print(f"{case}\t{wall}\t{real}\t{exit_code}")
PY

echo "[stage4_2B] outputs in $OUTDIR"
