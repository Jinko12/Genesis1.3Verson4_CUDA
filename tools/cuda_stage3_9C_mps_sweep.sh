#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <genesis4> <input.in> <ranks_per_gpu_list> [outdir]" >&2
  echo "Example: CUDA_VISIBLE_DEVICES=0,1 $0 ./build-cuda/genesis4 examples/.../Example4_a.cuda_profile.in 4,8,12,16" >&2
  exit 2
fi

EXE=$1
INPUT=$2
RPG_LIST=$3
OUTDIR=${4:-stage3_9C_mps_sweep_$(date +%Y%m%d_%H%M%S)}
MPIRUN=${MPIRUN:-mpirun}
mkdir -p "$OUTDIR"
summary="$OUTDIR/summary.tsv"
printf "mode\tranks_per_gpu\ttotal_ranks\twall_s\telapsed\tsys_s\texit\n" > "$summary"

visible_gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    python3 - <<'PY'
import os
v=os.environ.get('CUDA_VISIBLE_DEVICES','')
items=[x for x in v.split(',') if x.strip()!='']
print(len(items) if items else 1)
PY
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | wc -l
  else
    echo 1
  fi
}

start_mps() {
  export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps-$USER}
  export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log-$USER}
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "nvidia-cuda-mps-control not found" >&2
    return 1
  fi
  nvidia-cuda-mps-control -d
}

stop_mps() {
  if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
  fi
}

run_one() {
  local mode=$1 rpg=$2
  local ngpu
  ngpu=$(visible_gpu_count)
  local np=$((rpg * ngpu))
  local dir="$OUTDIR/${mode}_rpg_${rpg}_np_${np}"
  mkdir -p "$dir"
  local log="$dir/run.log"
  echo "[stage3.9C] mode=$mode ranks/GPU=$rpg total_ranks=$np visible_gpus=$ngpu"
  set +e
  GENESIS_CUDA_WORKER_RANKS_PER_DEVICE="$rpg" \
  GENESIS_CUDA_MAX_RANKS_PER_DEVICE="$rpg" \
  GENESIS_CUDA_DEVICE_POLICY="local_rank" \
  GENESIS_CUDA_PRINT_DEVICE_SUMMARY=1 \
  /usr/bin/time -p "$MPIRUN" -np "$np" "$EXE" "$INPUT" > "$log" 2>&1
  local code=$?
  set -e
  local wall elapsed sysv
  wall=$(awk '/Total Wall Clock Time/ {v=$NF} END {print v+0}' "$log")
  elapsed=$(awk '/^real / {print $2}' "$log" | tail -1)
  sysv=$(awk '/^sys / {print $2}' "$log" | tail -1)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$mode" "$rpg" "$np" "$wall" "$elapsed" "$sysv" "$code" >> "$summary"
}

IFS=',' read -r -a rpgs <<< "$RPG_LIST"
for rpg in "${rpgs[@]}"; do
  run_one "mps_off" "$rpg"
done

if start_mps; then
  trap stop_mps EXIT
  sleep 1
  for rpg in "${rpgs[@]}"; do
    run_one "mps_on" "$rpg"
  done
  stop_mps
  trap - EXIT
else
  echo "[stage3.9C] MPS unavailable; only mps_off cases were run" >&2
fi

echo "[stage3.9C] summary: $summary"
