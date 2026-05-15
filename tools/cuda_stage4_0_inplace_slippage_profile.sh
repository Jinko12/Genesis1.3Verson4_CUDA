#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 /path/to/genesis4 input.in mpi_ranks [output_dir]" >&2
  exit 2
fi

EXE=$1
INPUT=$2
NP=$3
OUTROOT=${4:-stage4_0_inplace_slippage_profile_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTROOT"

run_case() {
  local name=$1
  local inplace=$2
  local dir="$OUTROOT/$name"
  mkdir -p "$dir"
  (
    cd "$dir"
    echo "CASE=$name" | tee run.log
    echo "GENESIS_CUDA_INPLACE_SLIPPAGE=$inplace" | tee -a run.log
    export GENESIS_CUDA_INPLACE_SLIPPAGE=$inplace
    export GENESIS_CUDA_DEFER_FIELD_D2H=${GENESIS_CUDA_DEFER_FIELD_D2H:-1}
    export GENESIS_CUDA_BIND_FFT_FIELD=${GENESIS_CUDA_BIND_FFT_FIELD:-1}
    export GENESIS_CUDA_MPI_SLIPPAGE=${GENESIS_CUDA_MPI_SLIPPAGE:-1}
    export GENESIS_CUDA_FAST_KERNELS=${GENESIS_CUDA_FAST_KERNELS:-1}
    export GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD=${GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD:-1}
    export GENESIS_CUDA_DEVICE_POLICY=${GENESIS_CUDA_DEVICE_POLICY:-local_rank}
    if command -v nsys >/dev/null 2>&1; then
      /usr/bin/time -v nsys profile \
        --force-overwrite=true \
        --trace=cuda,nvtx,mpi,osrt \
        --cuda-memory-usage=true \
        --stats=true \
        --output=nsys_${name} \
        mpirun -np "$NP" "$EXE" "$INPUT" 2>&1 | tee -a run.log
      if [[ -f nsys_${name}.nsys-rep ]] && command -v nsys >/dev/null 2>&1; then
        nsys export --force --type sqlite --output nsys_${name}.sqlite nsys_${name}.nsys-rep >/dev/null 2>&1 || true
      fi
    else
      /usr/bin/time -v mpirun -np "$NP" "$EXE" "$INPUT" 2>&1 | tee -a run.log
    fi
  )
}

run_case inplace_on 1
run_case inplace_off 0

echo "Wrote $OUTROOT"
