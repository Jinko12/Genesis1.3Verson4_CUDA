#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<'USAGE'
Usage: tools/cuda_stage3_8_rank_sweep.sh /path/to/genesis4 input.in ranks_csv [output_dir]

Examples:
  # Single-GPU sweep on GPU0.  All ranks share one visible GPU.
  CUDA_VISIBLE_DEVICES=0 tools/cuda_stage3_8_rank_sweep.sh ./build-cuda/genesis4 examples/Example4-HGHG/Example4_a.cuda_profile.in 1,2,4,8,16,24,32

  # Multi-GPU sweep using all visible GPUs, testing round-robin and block mapping.
  tools/cuda_stage3_8_rank_sweep.sh ./build-cuda/genesis4 examples/Example4-HGHG/Example4_a.cuda_profile.in 2,4,8,16,32

Environment:
  GENESIS_CUDA_STAGE38_MODES       comma list: single,round_robin,block,forced0 (default: auto)
  GENESIS_CUDA_STAGE38_NSYS        1 to wrap runs in nsys profile (default: 0)
  GENESIS_CUDA_RANKS_PER_DEVICE    optional cap for block policy
  GENESIS_CUDA_VERBOSE_DEVICE      default 1
USAGE
  exit 2
fi

GENESIS_BIN=$1
INPUT=$2
RANKS_CSV=$3
OUTDIR=${4:-stage3_8_rank_sweep_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTDIR"

export GENESIS_CUDA_DEFER_FIELD_D2H=${GENESIS_CUDA_DEFER_FIELD_D2H:-1}
export GENESIS_CUDA_BIND_FFT_FIELD=${GENESIS_CUDA_BIND_FFT_FIELD:-1}
export GENESIS_CUDA_DIAG_REDUCTION=${GENESIS_CUDA_DIAG_REDUCTION:-1}
export GENESIS_CUDA_MPI_SLIPPAGE=${GENESIS_CUDA_MPI_SLIPPAGE:-1}
export GENESIS_CUDA_FAST_KERNELS=${GENESIS_CUDA_FAST_KERNELS:-1}
export GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD=${GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD:-1}
export GENESIS_CUDA_SYMMETRIC_TRANSVERSE=${GENESIS_CUDA_SYMMETRIC_TRANSVERSE:-1}
export GENESIS_CUDA_DEVICE_BINDING=${GENESIS_CUDA_DEVICE_BINDING:-1}
export GENESIS_CUDA_VERBOSE_DEVICE=${GENESIS_CUDA_VERBOSE_DEVICE:-1}

visible_count=$(python3 - <<'PY'
import os
v=os.environ.get('CUDA_VISIBLE_DEVICES','')
if v.strip():
    print(len([x for x in v.split(',') if x.strip()]))
else:
    print(0)
PY
)

if [[ -n "${GENESIS_CUDA_STAGE38_MODES:-}" ]]; then
  MODES_CSV="$GENESIS_CUDA_STAGE38_MODES"
elif [[ "$visible_count" == "1" ]]; then
  MODES_CSV="single"
else
  MODES_CSV="round_robin,block,forced0"
fi

IFS=',' read -r -a RANKS <<< "$RANKS_CSV"
IFS=',' read -r -a MODES <<< "$MODES_CSV"

run_one() {
  local mode=$1
  local np=$2
  local dir="$OUTDIR/${mode}_np${np}"
  mkdir -p "$dir"

  unset GENESIS_CUDA_DEVICE || true
  case "$mode" in
    single)
      export GENESIS_CUDA_DEVICE_POLICY=single
      ;;
    round_robin)
      export GENESIS_CUDA_DEVICE_POLICY=local_rank
      ;;
    block)
      export GENESIS_CUDA_DEVICE_POLICY=local_block
      ;;
    forced0)
      export GENESIS_CUDA_DEVICE=0
      export GENESIS_CUDA_DEVICE_POLICY=local_rank
      ;;
    *)
      echo "Unknown mode: $mode" >&2
      return 2
      ;;
  esac

  {
    echo "=== stage3.8 mode=${mode} np=${np} ==="
    date
    echo "GENESIS_BIN=${GENESIS_BIN}"
    echo "INPUT=${INPUT}"
    env | grep -E '^(CUDA_VISIBLE_DEVICES|GENESIS_CUDA_)' | sort
    nvidia-smi -L || true
  } | tee "$dir/run.log"

  if [[ "${GENESIS_CUDA_STAGE38_NSYS:-0}" == "1" ]]; then
    /usr/bin/time -f 'ELAPSED=%E MAXRSS_KB=%M EXIT=%x' \
      nsys profile --force-overwrite=true --stats=true \
        --trace=cuda,nvtx,mpi,osrt \
        --cuda-memory-usage=true \
        -o "$dir/nsys-${mode}-np${np}" \
        mpirun -np "$np" "$GENESIS_BIN" "$INPUT" \
        2>&1 | tee -a "$dir/run.log"
  else
    /usr/bin/time -f 'ELAPSED=%E MAXRSS_KB=%M EXIT=%x' \
      mpirun -np "$np" "$GENESIS_BIN" "$INPUT" \
      2>&1 | tee -a "$dir/run.log"
  fi
}

for mode in "${MODES[@]}"; do
  for np in "${RANKS[@]}"; do
    run_one "$mode" "$np"
  done
done

python3 "$(dirname "$0")/cuda_stage3_8_parse_logs.py" "$OUTDIR" > "$OUTDIR/summary.tsv" || true
cat "$OUTDIR/summary.tsv" || true

cat > "$OUTDIR/README.txt" <<TXT
Stage 3.8 rank/GPU sweep.

Modes:
  single      all ranks use visible device 0; use CUDA_VISIBLE_DEVICES=0 for strict single-GPU runs.
  round_robin local_rank % visible_device_count; good default multi-GPU balancing.
  block       contiguous local rank blocks per GPU; often better for NUMA/MPI-neighbor locality.
  forced0     regression mode; all ranks forced to device 0 even if multiple GPUs are visible.

Use summary.tsv to compare wall_s and device_counts.
TXT
