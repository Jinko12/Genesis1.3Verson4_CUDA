#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<'USAGE'
Usage: tools/cuda_stage3_9_worker_launch.sh /path/to/genesis4 input.in ranks_per_gpu [output_dir]

Runs Genesis CUDA with a per-GPU worker budget:
  MPI ranks = visible_gpu_count * ranks_per_gpu

Examples:
  # Single GPU, 8 ranks on the visible GPU.
  CUDA_VISIBLE_DEVICES=0 tools/cuda_stage3_9_worker_launch.sh ./build-cuda/genesis4 examples/Example4-HGHG/Example4_a.cuda_profile.in 8

  # Two visible GPUs, 4 ranks per GPU => 8 MPI ranks total.
  CUDA_VISIBLE_DEVICES=0,1 tools/cuda_stage3_9_worker_launch.sh ./build-cuda/genesis4 examples/Example4-HGHG/Example4_a.cuda_profile.in 4

Environment:
  GENESIS_CUDA_DEVICE_POLICY        default local_rank, i.e. round-robin across visible GPUs
  GENESIS_CUDA_MAX_RANKS_PER_DEVICE default ranks_per_gpu for warning/strict guard
  GENESIS_CUDA_STRICT_RANK_BUDGET   set 1 to abort if mapping exceeds budget
  GENESIS_CUDA_STAGE39_NSYS         set 1 to wrap run with nsys profile
USAGE
  exit 2
fi

GENESIS_BIN=$1
INPUT=$2
RANKS_PER_GPU=$3
OUTDIR=${4:-stage3_9_worker_launch_rpg${RANKS_PER_GPU}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTDIR"

visible_gpu_count() {
  python3 - <<'PY'
import os, subprocess
v = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
if v:
    items = [x for x in v.split(',') if x.strip()]
    print(max(1, len(items)))
else:
    try:
        out = subprocess.check_output(['nvidia-smi','-L'], text=True, stderr=subprocess.DEVNULL)
        n = sum(1 for line in out.splitlines() if line.strip().startswith('GPU '))
        print(max(1, n))
    except Exception:
        print(1)
PY
}

GPU_COUNT=$(visible_gpu_count)
NP=$((GPU_COUNT * RANKS_PER_GPU))

export GENESIS_CUDA_DEFER_FIELD_D2H=${GENESIS_CUDA_DEFER_FIELD_D2H:-1}
export GENESIS_CUDA_BIND_FFT_FIELD=${GENESIS_CUDA_BIND_FFT_FIELD:-1}
export GENESIS_CUDA_DIAG_REDUCTION=${GENESIS_CUDA_DIAG_REDUCTION:-1}
export GENESIS_CUDA_MPI_SLIPPAGE=${GENESIS_CUDA_MPI_SLIPPAGE:-1}
export GENESIS_CUDA_FAST_KERNELS=${GENESIS_CUDA_FAST_KERNELS:-1}
export GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD=${GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD:-1}
export GENESIS_CUDA_SYMMETRIC_TRANSVERSE=${GENESIS_CUDA_SYMMETRIC_TRANSVERSE:-1}
export GENESIS_CUDA_DEVICE_BINDING=${GENESIS_CUDA_DEVICE_BINDING:-1}
export GENESIS_CUDA_DEVICE_POLICY=${GENESIS_CUDA_DEVICE_POLICY:-local_rank}
export GENESIS_CUDA_PRINT_DEVICE_SUMMARY=${GENESIS_CUDA_PRINT_DEVICE_SUMMARY:-1}
export GENESIS_CUDA_VERBOSE_DEVICE=${GENESIS_CUDA_VERBOSE_DEVICE:-0}
export GENESIS_CUDA_MAX_RANKS_PER_DEVICE=${GENESIS_CUDA_MAX_RANKS_PER_DEVICE:-$RANKS_PER_GPU}

LOG="$OUTDIR/run.log"
{
  echo "=== Stage 3.9 per-GPU worker launch ==="
  date
  echo "GENESIS_BIN=${GENESIS_BIN}"
  echo "INPUT=${INPUT}"
  echo "visible_gpu_count=${GPU_COUNT}"
  echo "ranks_per_gpu=${RANKS_PER_GPU}"
  echo "mpi_ranks=${NP}"
  env | grep -E '^(CUDA_VISIBLE_DEVICES|GENESIS_CUDA_)' | sort
  nvidia-smi -L || true
} | tee "$LOG"

if [[ "${GENESIS_CUDA_STAGE39_NSYS:-0}" == "1" ]]; then
  /usr/bin/time -f 'ELAPSED=%E MAXRSS_KB=%M EXIT=%x' \
    nsys profile --force-overwrite=true --stats=true \
      --trace=cuda,nvtx,mpi,osrt \
      --cuda-memory-usage=true \
      -o "$OUTDIR/nsys-stage3-9-rpg${RANKS_PER_GPU}" \
      mpirun -np "$NP" "$GENESIS_BIN" "$INPUT" \
      2>&1 | tee -a "$LOG"
else
  /usr/bin/time -f 'ELAPSED=%E MAXRSS_KB=%M EXIT=%x' \
    mpirun -np "$NP" "$GENESIS_BIN" "$INPUT" \
    2>&1 | tee -a "$LOG"
fi

python3 "$(dirname "$0")/cuda_stage3_9_parse_logs.py" "$LOG" > "$OUTDIR/summary.tsv" || true
cat "$OUTDIR/summary.tsv" || true
