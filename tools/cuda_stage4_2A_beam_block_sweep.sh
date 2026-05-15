#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<'USAGE'
Usage:
  tools/cuda_stage4_2A_beam_block_sweep.sh <genesis4> <input.in> <mpi_ranks> [outdir]

Purpose:
  Low-risk Stage 4.2A A/B sweep for CUDA launch block sizes. This script
  does not change physics settings; it only varies CUDA launch dimensions via
  runtime environment variables added in Stage 4.2A.

Environment overrides:
  GENESIS_STAGE4_2A_LONG_BLOCKS="128,192,256,320,384"
  GENESIS_STAGE4_2A_TRANS_BLOCKS="128,192,256,320,384"
  GENESIS_STAGE4_2A_SOURCE_BLOCKS="128,256,512"
  MPIRUN="mpirun"
USAGE
  exit 2
fi

EXE=$1
INPUT=$2
RANKS=$3
OUTDIR=${4:-stage4_2A_beam_block_sweep_$(date +%Y%m%d_%H%M%S)}
MPIRUN_CMD=${MPIRUN:-mpirun}

LONG_BLOCKS=${GENESIS_STAGE4_2A_LONG_BLOCKS:-128,192,256,320,384}
TRANS_BLOCKS=${GENESIS_STAGE4_2A_TRANS_BLOCKS:-128,192,256,320,384}
SOURCE_BLOCKS=${GENESIS_STAGE4_2A_SOURCE_BLOCKS:-128,256,512}

mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.tsv"
printf "case\tlong_block\ttrans_block\tsource_block\twall_clock_s\texit\n" > "$SUMMARY"

parse_wall() {
  local log=$1
  awk '/Total Wall Clock Time/ {val=$NF} END {if (val != "") print val; else print "NA"}' "$log"
}

run_case() {
  local name=$1
  local long_block=$2
  local trans_block=$3
  local source_block=$4
  local dir="$OUTDIR/$name"
  local log="$dir/run.log"
  mkdir -p "$dir"
  echo "[Stage4.2A] case=$name long=$long_block trans=$trans_block source=$source_block"
  set +e
  (
    export GENESIS_CUDA_BEAM_LONGITUDINAL_BLOCK="$long_block"
    export GENESIS_CUDA_BEAM_TRANSVERSE_BLOCK="$trans_block"
    export GENESIS_CUDA_SOURCE_DEPOSITION_BLOCK="$source_block"
    export GENESIS_CUDA_INPLACE_SLIPPAGE=${GENESIS_CUDA_INPLACE_SLIPPAGE:-1}
    export GENESIS_CUDA_FAST_KERNELS=${GENESIS_CUDA_FAST_KERNELS:-1}
    export GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD=${GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD:-1}
    export GENESIS_CUDA_SYMMETRIC_TRANSVERSE=${GENESIS_CUDA_SYMMETRIC_TRANSVERSE:-1}
    /usr/bin/time -p $MPIRUN_CMD -np "$RANKS" "$EXE" "$INPUT"
  ) > "$log" 2>&1
  local status=$?
  set -e
  local wall
  wall=$(parse_wall "$log")
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$long_block" "$trans_block" "$source_block" "$wall" "$status" >> "$SUMMARY"
}

run_case baseline_256 256 256 256

IFS=',' read -ra LBS <<< "$LONG_BLOCKS"
for b in "${LBS[@]}"; do
  [[ "$b" == "256" ]] && continue
  run_case "long_${b}" "$b" 256 256
done

IFS=',' read -ra TBS <<< "$TRANS_BLOCKS"
for b in "${TBS[@]}"; do
  [[ "$b" == "256" ]] && continue
  run_case "trans_${b}" 256 "$b" 256
done

IFS=',' read -ra SBS <<< "$SOURCE_BLOCKS"
for b in "${SBS[@]}"; do
  [[ "$b" == "256" ]] && continue
  run_case "source_${b}" 256 256 "$b"
done

python3 - <<PY
from pathlib import Path
summary = Path('$SUMMARY')
rows = []
for line in summary.read_text().strip().splitlines()[1:]:
    case, lb, tb, sb, wall, status = line.split('\t')
    try:
        w = float(wall)
    except ValueError:
        continue
    if status == '0':
        rows.append((w, case, lb, tb, sb))
if rows:
    rows.sort()
    w, case, lb, tb, sb = rows[0]
    rec = Path('$OUTDIR') / 'recommendation.txt'
    rec.write_text(
        f"best_case={case}\nwall_clock_s={w}\n"
        f"GENESIS_CUDA_BEAM_LONGITUDINAL_BLOCK={lb}\n"
        f"GENESIS_CUDA_BEAM_TRANSVERSE_BLOCK={tb}\n"
        f"GENESIS_CUDA_SOURCE_DEPOSITION_BLOCK={sb}\n"
    )
    print(rec.read_text())
else:
    print('No successful cases found')
PY

echo "Summary: $SUMMARY"
