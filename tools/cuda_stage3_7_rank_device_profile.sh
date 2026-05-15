#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 /path/to/genesis4 /path/to/input.in MPI_RANKS [output_dir]" >&2
  exit 2
fi

GENESIS_BIN=$1
INPUT=$2
RANKS=$3
OUTDIR=${4:-stage3_7_rank_device_profile_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTDIR"

export GENESIS_CUDA_NVTX=${GENESIS_CUDA_NVTX:-1}
export GENESIS_CUDA_DEVICE_BINDING=${GENESIS_CUDA_DEVICE_BINDING:-1}
export GENESIS_CUDA_DEVICE_POLICY=${GENESIS_CUDA_DEVICE_POLICY:-local_rank}
export GENESIS_CUDA_VERBOSE_DEVICE=${GENESIS_CUDA_VERBOSE_DEVICE:-1}

run_case() {
  local label=$1
  shift
  local dir="$OUTDIR/$label"
  mkdir -p "$dir"
  echo "=== $label ===" | tee "$dir/run.log"
  env | grep '^GENESIS_CUDA_' | sort | tee -a "$dir/run.log"
  nvidia-smi -L | tee -a "$dir/run.log" || true
  /usr/bin/time -f 'ELAPSED=%E MAXRSS_KB=%M EXIT=%x' \
    nsys profile --force-overwrite=true --stats=true \
      --trace=cuda,nvtx,mpi,osrt \
      --cuda-memory-usage=true \
      -o "$dir/nsys-$label" \
      mpirun -np "$RANKS" "$GENESIS_BIN" "$INPUT" \
      2>&1 | tee -a "$dir/run.log"
}

# Balanced default: local rank modulo visible device count.
export GENESIS_CUDA_DEVICE_BINDING=1
export GENESIS_CUDA_DEVICE_POLICY=local_rank
unset GENESIS_CUDA_DEVICE || true
run_case rank-device-balanced

# Regression/debug case: force all ranks to GPU0.
export GENESIS_CUDA_DEVICE_BINDING=1
export GENESIS_CUDA_DEVICE=0
run_case rank-device-gpu0-forced

cat > "$OUTDIR/README.txt" <<TXT
Generated rank-to-device profile cases:
  rank-device-balanced: default local_rank % cudaGetDeviceCount() mapping
  rank-device-gpu0-forced: conservative regression case; all ranks on GPU0

Expected result on a 2-GPU single node with 32 ranks:
  balanced -> 16 ranks per GPU, GPU0 and GPU1 both active
  forced   -> 32 ranks on GPU0, GPU1 mostly idle
TXT
