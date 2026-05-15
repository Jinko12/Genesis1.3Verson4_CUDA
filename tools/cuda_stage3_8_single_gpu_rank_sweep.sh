#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 /path/to/genesis4 input.in [np_list] [output_dir]" >&2
  echo "Example: CUDA_VISIBLE_DEVICES=0 $0 ./build-cuda/genesis4 examples/Example4-HGHG/Example4_a.cuda_profile.in '1 2 4 8 16 24 32'" >&2
  exit 2
fi

GENESIS_BIN=$1
INPUT=$2
NP_LIST=${3:-"1 2 4 8 16 24 32"}
OUTDIR=${4:-stage3_8_single_gpu_rank_sweep_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTDIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export GENESIS_CUDA_DEVICE_BINDING=${GENESIS_CUDA_DEVICE_BINDING:-1}
export GENESIS_CUDA_DEVICE_POLICY=${GENESIS_CUDA_DEVICE_POLICY:-local_rank}
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
echo -e "np\twall_clock_s\telapsed\tmaxrss_kb\texit_code\tlog" > "$summary"

run_np() {
  local np=$1
  local dir="$OUTDIR/np${np}"
  mkdir -p "$dir"
  echo "=== single-gpu np=${np} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ===" | tee "$dir/run.log"
  env | grep -E '^(CUDA_VISIBLE_DEVICES|GENESIS_CUDA_)' | sort | tee -a "$dir/run.log"
  nvidia-smi -L | tee -a "$dir/run.log" || true
  set +e
  /usr/bin/time -f 'ELAPSED=%E MAXRSS_KB=%M EXIT=%x' \
    mpirun -np "$np" "$GENESIS_BIN" "$INPUT" \
    2>&1 | tee -a "$dir/run.log"
  local code=${PIPESTATUS[0]}
  set -e
  local wall elapsed maxrss exit_code
  wall=$(awk '/Total Wall Clock Time:/ {v=$(NF-1)} END{print v+0}' "$dir/run.log")
  elapsed=$(awk -F'[ =]' '/ELAPSED=/ {for(i=1;i<=NF;i++) if($i=="ELAPSED") print $(i+1)}' "$dir/run.log" | tail -1)
  maxrss=$(awk -F'[ =]' '/MAXRSS_KB=/ {for(i=1;i<=NF;i++) if($i=="MAXRSS_KB") print $(i+1)}' "$dir/run.log" | tail -1)
  exit_code=$(awk -F'[ =]' '/EXIT=/ {for(i=1;i<=NF;i++) if($i=="EXIT") print $(i+1)}' "$dir/run.log" | tail -1)
  echo -e "${np}\t${wall}\t${elapsed:-NA}\t${maxrss:-NA}\t${exit_code:-$code}\t${dir}/run.log" >> "$summary"
}

for np in ${NP_LIST}; do
  run_np "$np"
done

cat <<EOF
Generated single-GPU rank sweep:
  $OUTDIR
Summary:
  $summary

Recommended follow-up:
  1. Pick the fastest np that does not OOM.
  2. Repeat that np with Nsight Systems if wall time changes unexpectedly.
  3. Compare with MPS on/off only after this baseline is stable.
EOF
