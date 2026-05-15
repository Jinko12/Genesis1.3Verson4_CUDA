#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 /path/to/genesis4 input.in MPI_RANKS [output_dir]" >&2
  echo "Policies tested: local_rank round-robin, block contiguous mapping, forced GPU0." >&2
  exit 2
fi

GENESIS_BIN=$1
INPUT=$2
RANKS=$3
OUTDIR=${4:-stage3_8_multi_gpu_policy_profile_np${RANKS}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTDIR"

export GENESIS_CUDA_DEVICE_BINDING=1
export GENESIS_CUDA_VERBOSE_DEVICE=${GENESIS_CUDA_VERBOSE_DEVICE:-1}
export GENESIS_CUDA_NVTX=${GENESIS_CUDA_NVTX:-1}
export GENESIS_CUDA_DEFER_FIELD_D2H=${GENESIS_CUDA_DEFER_FIELD_D2H:-1}
export GENESIS_CUDA_BIND_FFT_FIELD=${GENESIS_CUDA_BIND_FFT_FIELD:-1}
export GENESIS_CUDA_DIAG_REDUCTION=${GENESIS_CUDA_DIAG_REDUCTION:-1}
export GENESIS_CUDA_MPI_SLIPPAGE=${GENESIS_CUDA_MPI_SLIPPAGE:-1}
export GENESIS_CUDA_FAST_KERNELS=${GENESIS_CUDA_FAST_KERNELS:-1}
export GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD=${GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD:-1}
export GENESIS_CUDA_SYMMETRIC_TRANSVERSE=${GENESIS_CUDA_SYMMETRIC_TRANSVERSE:-1}

summary="$OUTDIR/summary.tsv"
echo -e "case\tpolicy\tvisible_devices\twall_clock_s\telapsed\tmaxrss_kb\texit_code\tlog" > "$summary"

run_case() {
  local label=$1
  local policy=$2
  local forced=${3:-}
  local dir="$OUTDIR/$label"
  mkdir -p "$dir"
  export GENESIS_CUDA_DEVICE_POLICY="$policy"
  if [[ -n "$forced" ]]; then
    export GENESIS_CUDA_DEVICE="$forced"
  else
    unset GENESIS_CUDA_DEVICE || true
  fi
  unset GENESIS_CUDA_RANKS_PER_DEVICE || true
  echo "=== ${label}: policy=${policy} forced=${forced:-none} ranks=${RANKS} ===" | tee "$dir/run.log"
  env | grep -E '^(CUDA_VISIBLE_DEVICES|GENESIS_CUDA_)' | sort | tee -a "$dir/run.log"
  nvidia-smi -L | tee -a "$dir/run.log" || true
  set +e
  /usr/bin/time -f 'ELAPSED=%E MAXRSS_KB=%M EXIT=%x' \
    nsys profile --force-overwrite=true --stats=true \
      --trace=cuda,nvtx,mpi,osrt \
      --cuda-memory-usage=true \
      -o "$dir/nsys-$label" \
      mpirun -np "$RANKS" "$GENESIS_BIN" "$INPUT" \
      2>&1 | tee -a "$dir/run.log"
  local code=${PIPESTATUS[0]}
  set -e
  local wall elapsed maxrss exit_code visible
  visible=${CUDA_VISIBLE_DEVICES:-all}
  wall=$(awk '/Total Wall Clock Time:/ {v=$(NF-1)} END{print v+0}' "$dir/run.log")
  elapsed=$(awk -F'[ =]' '/ELAPSED=/ {for(i=1;i<=NF;i++) if($i=="ELAPSED") print $(i+1)}' "$dir/run.log" | tail -1)
  maxrss=$(awk -F'[ =]' '/MAXRSS_KB=/ {for(i=1;i<=NF;i++) if($i=="MAXRSS_KB") print $(i+1)}' "$dir/run.log" | tail -1)
  exit_code=$(awk -F'[ =]' '/EXIT=/ {for(i=1;i<=NF;i++) if($i=="EXIT") print $(i+1)}' "$dir/run.log" | tail -1)
  echo -e "${label}\t${policy}\t${visible}\t${wall}\t${elapsed:-NA}\t${maxrss:-NA}\t${exit_code:-$code}\t${dir}/run.log" >> "$summary"
}

# Round-robin mapping: ranks 0,2,4... to GPU0 and 1,3,5... to GPU1 on a 2-GPU node.
run_case round_robin local_rank

# Block mapping: contiguous rank blocks per GPU. This may reduce cross-GPU neighbor boundary traffic.
run_case block block

# Conservative regression: all ranks forced to visible device 0.
run_case forced_gpu0 single 0

cat <<EOF
Generated multi-GPU policy profile:
  $OUTDIR
Summary:
  $summary

Interpretation:
  round_robin: validated Stage 3.7 default mapping.
  block      : contiguous rank blocks per GPU; often better for nearest-neighbor slippage/MPI boundaries.
  forced_gpu0: regression/debug baseline.
EOF
