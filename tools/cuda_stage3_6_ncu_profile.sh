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
OUTROOT="${OUTROOT:-stage3_6_ncu_${STAMP}}"
mkdir -p "${OUTROOT}"

export GENESIS_CUDA_DEFER_FIELD_D2H="${GENESIS_CUDA_DEFER_FIELD_D2H:-1}"
export GENESIS_CUDA_BIND_FFT_FIELD="${GENESIS_CUDA_BIND_FFT_FIELD:-1}"
export GENESIS_CUDA_DIAG_REDUCTION="${GENESIS_CUDA_DIAG_REDUCTION:-1}"
export GENESIS_CUDA_MPI_SLIPPAGE="${GENESIS_CUDA_MPI_SLIPPAGE:-1}"
export GENESIS_CUDA_FAST_KERNELS="${GENESIS_CUDA_FAST_KERNELS:-1}"
export GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD="${GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD:-1}"
export GENESIS_CUDA_SYMMETRIC_TRANSVERSE="${GENESIS_CUDA_SYMMETRIC_TRANSVERSE:-1}"
export GENESIS_CUDA_NVTX="${GENESIS_CUDA_NVTX:-1}"

# Override this when testing one kernel at a time.  Example:
#   NCU_KERNEL_REGEX='beamLongitudinalOneFieldCachedInterpKernel' tools/...
KERNEL_REGEX="${NCU_KERNEL_REGEX:-beamLongitudinalOneFieldCachedInterpKernel|beamTrackTransverse.*|buildSourceFromSoAKernel|beamBunchingDiagnosticKernel|beamMomentsDiagnosticKernel}"
NCU_SECTIONS="${NCU_SECTIONS:-SpeedOfLight,LaunchStats,Occupancy,MemoryWorkloadAnalysis,SchedulerStats,WarpStateStats,SourceCounters}"

run_cmd=("${GENESIS_BIN}" "${INPUT_FILE}")
if [[ "${MPI_RANKS}" -gt 1 ]]; then
  run_cmd=(mpirun -np "${MPI_RANKS}" "${GENESIS_BIN}" "${INPUT_FILE}")
fi

cat <<EOF2
[stage3.6-ncu]
  output       : ${OUTROOT}
  ranks        : ${MPI_RANKS}
  kernel regex : ${KERNEL_REGEX}
  sections     : ${NCU_SECTIONS}
EOF2

ncu \
  --target-processes all \
  --kernel-name "regex:${KERNEL_REGEX}" \
  --section "${NCU_SECTIONS}" \
  --force-overwrite \
  -o "${OUTROOT}/stage3_6_hotkernels" \
  "${run_cmd[@]}" | tee "${OUTROOT}/stage3_6_hotkernels.log"

# CSV makes it easier to grep bottleneck metrics in batch jobs.
ncu \
  --target-processes all \
  --kernel-name "regex:${KERNEL_REGEX}" \
  --section "${NCU_SECTIONS}" \
  --csv \
  --page raw \
  "${run_cmd[@]}" > "${OUTROOT}/stage3_6_hotkernels_raw.csv" 2> "${OUTROOT}/stage3_6_hotkernels_raw.err" || true

cat <<EOF3
Generated:
  ${OUTROOT}/stage3_6_hotkernels.ncu-rep
  ${OUTROOT}/stage3_6_hotkernels.log
  ${OUTROOT}/stage3_6_hotkernels_raw.csv
EOF3
