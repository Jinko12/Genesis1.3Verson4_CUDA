#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 /path/to/genesis4 input.in [mpi_ranks]" >&2
  exit 2
fi

GENESIS_BIN="$1"
INPUT_FILE="$2"
MPI_RANKS="${3:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTROOT="${OUTROOT:-stage5_interp_cache_profile_${STAMP}}"
mkdir -p "${OUTROOT}"

run_case() {
  local label="$1"
  local cache="$2"
  local outdir="${OUTROOT}/${label}"
  mkdir -p "${outdir}"
  echo "[stage5-interp-cache-profile] ${label}: GENESIS_CUDA_FAST_KERNELS=1, GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD=${cache}, ranks=${MPI_RANKS}"
  (
    cd "${outdir}"
    export GENESIS_CUDA_DEFER_FIELD_D2H=1
    export GENESIS_CUDA_BIND_FFT_FIELD=1
    export GENESIS_CUDA_DIAG_REDUCTION=1
    export GENESIS_CUDA_MPI_SLIPPAGE=1
    export GENESIS_CUDA_FAST_KERNELS=1
    export GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD="${cache}"
    if [[ "${MPI_RANKS}" -gt 1 ]]; then
      nsys profile --trace=cuda,mpi,osrt --stats=true --force-overwrite=true \
        -o "nsys-${label}" mpirun -np "${MPI_RANKS}" "${GENESIS_BIN}" "${INPUT_FILE}"
    else
      nsys profile --trace=cuda,osrt --stats=true --force-overwrite=true \
        -o "nsys-${label}" "${GENESIS_BIN}" "${INPUT_FILE}"
    fi
  ) | tee "${outdir}.log"
}

run_case "interp-cache-on" 1
run_case "interp-cache-off" 0

cat <<EOF2

Generated profiles under: ${OUTROOT}
Compare cuda_gpu_kern_sum for:
  - beamLongitudinalOneFieldCachedInterpKernel vs beamLongitudinalOneFieldKernel
  - total wall time and CUDA kernel total time
  - H2D/D2H/D2D totals to confirm no communication regression
EOF2
