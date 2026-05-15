#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <genesis4> <input.in> [ranks_per_gpu_csv]" >&2
  echo "Example: CUDA_VISIBLE_DEVICES=0,1 $0 ./build-cuda/genesis4 examples/Example4-HGHG/Example4_a.cuda_profile.in 1,2,4,8" >&2
  exit 2
fi

GENESIS_BIN=$1
INPUT=$2
RANKS_PER_GPU_CSV=${3:-1,2,4,8,12,16}
OUTDIR=${OUTDIR:-stage3_9B_memory_audit_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTDIR"

IFS=',' read -r -a RANKS_PER_GPU_LIST <<< "$RANKS_PER_GPU_CSV"

VISIBLE=${CUDA_VISIBLE_DEVICES:-}
if [[ -n "$VISIBLE" ]]; then
  IFS=',' read -r -a DEVICES <<< "$VISIBLE"
  NDEV=${#DEVICES[@]}
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    NDEV=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
  else
    NDEV=1
  fi
fi
if [[ "$NDEV" -lt 1 ]]; then NDEV=1; fi

echo "# outdir=$OUTDIR" | tee "$OUTDIR/summary.tsv"
echo -e "ranks_per_gpu\ttotal_ranks\texit\twall_s\tpeak_mib_max\tpeak_mib_sum\tcufft_workspace_peak_mib_sum\tlog" | tee -a "$OUTDIR/summary.tsv"

for rpg in "${RANKS_PER_GPU_LIST[@]}"; do
  total=$((rpg * NDEV))
  case_dir="$OUTDIR/rpg_${rpg}_np_${total}"
  mkdir -p "$case_dir"
  log="$case_dir/run.log"
  echo "[Stage 3.9B] ranks_per_gpu=$rpg total_ranks=$total visible_gpus=$NDEV" | tee "$log"
  set +e
  GENESIS_CUDA_MEMORY_AUDIT=1 \
  GENESIS_CUDA_MEMORY_AUDIT_TOP=${GENESIS_CUDA_MEMORY_AUDIT_TOP:-10} \
  GENESIS_CUDA_LAZY_PARTICLE_STAGING=${GENESIS_CUDA_LAZY_PARTICLE_STAGING:-1} \
  GENESIS_CUDA_WORKER_RANKS_PER_DEVICE=$rpg \
  GENESIS_CUDA_MAX_RANKS_PER_DEVICE=$rpg \
  GENESIS_CUDA_DEVICE_POLICY=${GENESIS_CUDA_DEVICE_POLICY:-local_rank} \
  GENESIS_CUDA_PRINT_DEVICE_SUMMARY=1 \
  /usr/bin/time -p mpirun -np "$total" "$GENESIS_BIN" "$INPUT" >> "$log" 2>&1
  rc=$?
  set -e
  python3 tools/cuda_stage3_9B_parse_memory_audit.py "$log" --tsv >> "$OUTDIR/summary.tsv" || true
  if [[ "$rc" -ne 0 ]]; then
    echo "[Stage 3.9B] case failed: ranks_per_gpu=$rpg rc=$rc" | tee -a "$log"
  fi
done

python3 tools/cuda_stage3_9B_parse_memory_audit.py "$OUTDIR" --recommend > "$OUTDIR/recommendation.txt" || true
cat "$OUTDIR/recommendation.txt" || true
