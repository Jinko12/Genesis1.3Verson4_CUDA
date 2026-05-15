#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 /path/to/genesis4 input.in [mpi_ranks]" >&2
  exit 2
fi

GENESIS_BIN="$1"
INPUT_FILE="$2"
MPI_RANKS="${3:-4}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTROOT="${OUTROOT:-stage3_6_nvtx_ncu_${STAMP}}"
mkdir -p "${OUTROOT}"

export GENESIS_CUDA_DEFER_FIELD_D2H="${GENESIS_CUDA_DEFER_FIELD_D2H:-1}"
export GENESIS_CUDA_BIND_FFT_FIELD="${GENESIS_CUDA_BIND_FFT_FIELD:-1}"
export GENESIS_CUDA_DIAG_REDUCTION="${GENESIS_CUDA_DIAG_REDUCTION:-1}"
export GENESIS_CUDA_MPI_SLIPPAGE="${GENESIS_CUDA_MPI_SLIPPAGE:-1}"
export GENESIS_CUDA_FAST_KERNELS="${GENESIS_CUDA_FAST_KERNELS:-1}"
export GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD="${GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD:-1}"
export GENESIS_CUDA_SYMMETRIC_TRANSVERSE="${GENESIS_CUDA_SYMMETRIC_TRANSVERSE:-1}"
export GENESIS_CUDA_NVTX="${GENESIS_CUDA_NVTX:-1}"

run_cmd=("${GENESIS_BIN}" "${INPUT_FILE}")
if [[ "${MPI_RANKS}" -gt 1 ]]; then
  run_cmd=(mpirun -np "${MPI_RANKS}" "${GENESIS_BIN}" "${INPUT_FILE}")
fi

cat <<EOF2
[stage3.6-profile]
  output : ${OUTROOT}
  ranks  : ${MPI_RANKS}
  input  : ${INPUT_FILE}
  nvtx   : ${GENESIS_CUDA_NVTX}
EOF2

nsys profile \
  --trace=cuda,nvtx,mpi,osrt \
  --cuda-memory-usage=true \
  --stats=true \
  --force-overwrite=true \
  -o "${OUTROOT}/stage3_6_nsys" \
  "${run_cmd[@]}" | tee "${OUTROOT}/stage3_6_nsys.log"

if [[ -f "${OUTROOT}/stage3_6_nsys.nsys-rep" ]]; then
  nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,nvtx_sum,mpi_event_sum \
    "${OUTROOT}/stage3_6_nsys.nsys-rep" \
    > "${OUTROOT}/stage3_6_nsys_stats.txt" || true
fi

if [[ -f "${OUTROOT}/stage3_6_nsys.sqlite" ]]; then
  python3 "$(dirname "$0")/cuda_stage3_6_sqlite_summary.py" \
    "${OUTROOT}/stage3_6_nsys.sqlite" \
    > "${OUTROOT}/stage3_6_sqlite_summary.txt" || true
fi

cat <<EOF3
Generated profile directory:
  ${OUTROOT}

Next kernel-level step:
  tools/cuda_stage3_6_ncu_profile.sh ${GENESIS_BIN} ${INPUT_FILE} ${MPI_RANKS}
EOF3
