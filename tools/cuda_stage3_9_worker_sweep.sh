#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<'USAGE'
Usage: tools/cuda_stage3_9_worker_sweep.sh /path/to/genesis4 input.in ranks_per_gpu_csv [output_dir]

Sweeps ranks-per-GPU worker budgets. This is the Stage 3.9 production-oriented
replacement for arbitrary MPI-rank sweeps: it compares 1,2,4,8,... ranks per
visible GPU rather than raw MPI ranks.

Examples:
  CUDA_VISIBLE_DEVICES=0 tools/cuda_stage3_9_worker_sweep.sh ./build-cuda/genesis4 examples/Example4-HGHG/Example4_a.cuda_profile.in 1,2,4,8,16
  CUDA_VISIBLE_DEVICES=0,1 tools/cuda_stage3_9_worker_sweep.sh ./build-cuda/genesis4 examples/Example4-HGHG/Example4_a.cuda_profile.in 1,2,4,8,16
USAGE
  exit 2
fi

GENESIS_BIN=$1
INPUT=$2
RPG_CSV=$3
OUTDIR=${4:-stage3_9_worker_sweep_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTDIR"

IFS=',' read -r -a RPGS <<< "$RPG_CSV"
for rpg in "${RPGS[@]}"; do
  rpg_trim=$(echo "$rpg" | xargs)
  [[ -z "$rpg_trim" ]] && continue
  "$(dirname "$0")/cuda_stage3_9_worker_launch.sh" \
    "$GENESIS_BIN" "$INPUT" "$rpg_trim" "$OUTDIR/rpg${rpg_trim}"
done

python3 "$(dirname "$0")/cuda_stage3_9_parse_logs.py" "$OUTDIR" > "$OUTDIR/summary.tsv" || true
cat "$OUTDIR/summary.tsv" || true
python3 "$(dirname "$0")/cuda_stage3_9_recommend.py" "$OUTDIR/summary.tsv" > "$OUTDIR/recommendation.txt" || true
cat "$OUTDIR/recommendation.txt" || true
