#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <genesis4> <input.in> <mpi_ranks> [outdir]" >&2
  exit 2
fi

GENESIS_BIN=$1
INPUT=$2
MPI_RANKS=$3
OUTDIR=${4:-stage4_0A_inplace_slippage_profile_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$OUTDIR"
run_case() {
  local name=$1
  local inplace=$2
  local dir="$OUTDIR/$name"
  mkdir -p "$dir"
  (
    cd "$dir"
    export GENESIS_CUDA_INPLACE_SLIPPAGE=$inplace
    export GENESIS_CUDA_DEFER_FIELD_D2H=${GENESIS_CUDA_DEFER_FIELD_D2H:-1}
    export GENESIS_CUDA_BIND_FFT_FIELD=${GENESIS_CUDA_BIND_FFT_FIELD:-1}
    export GENESIS_CUDA_MPI_SLIPPAGE=${GENESIS_CUDA_MPI_SLIPPAGE:-1}
    export GENESIS_CUDA_FAST_KERNELS=${GENESIS_CUDA_FAST_KERNELS:-1}
    export GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD=${GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD:-1}
    export GENESIS_CUDA_DEVICE_POLICY=${GENESIS_CUDA_DEVICE_POLICY:-local_rank}
    export GENESIS_CUDA_PRINT_DEVICE_SUMMARY=${GENESIS_CUDA_PRINT_DEVICE_SUMMARY:-1}
    echo "CASE=$name GENESIS_CUDA_INPLACE_SLIPPAGE=$inplace" | tee run.log
    if command -v nsys >/dev/null 2>&1; then
      /usr/bin/time -v nsys profile \
        --trace=cuda,nvtx,mpi,osrt \
        --cuda-memory-usage=true \
        --stats=true \
        --force-overwrite=true \
        -o stage4_0A_${name} \
        mpirun -np "$MPI_RANKS" "$GENESIS_BIN" "$INPUT" \
        >> run.log 2>&1 || echo "EXIT=$?" >> run.log
      if [[ -f stage4_0A_${name}.nsys-rep ]]; then
        nsys export --type sqlite --force-overwrite=true \
          -o stage4_0A_${name}.sqlite stage4_0A_${name}.nsys-rep \
          >> run.log 2>&1 || true
      fi
    else
      /usr/bin/time -v mpirun -np "$MPI_RANKS" "$GENESIS_BIN" "$INPUT" \
        >> run.log 2>&1 || echo "EXIT=$?" >> run.log
    fi
  )
}

run_case inplace_on 1
run_case inplace_off 0

cat > "$OUTDIR/README.txt" <<EOF2
Stage 4.0A in-place slippage A/B profile

Cases:
  inplace_on  : GENESIS_CUDA_INPLACE_SLIPPAGE=1
  inplace_off : GENESIS_CUDA_INPLACE_SLIPPAGE=0

Compare:
  - Wall Clock / time real
  - D2D memcpy totals
  - largest D2H, to ensure no full field-record D2H returns
  - HDF5 correctness with tools/cuda_stage3_9B_correctness_guard.py
EOF2

echo "Wrote $OUTDIR"
