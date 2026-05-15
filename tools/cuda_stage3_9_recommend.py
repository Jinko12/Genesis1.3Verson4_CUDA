#!/usr/bin/env python3
import argparse
import csv
import math
import sys

ap = argparse.ArgumentParser(description='Recommend a Stage 3.9 ranks-per-GPU setting from summary.tsv')
ap.add_argument('summary')
args = ap.parse_args()
rows=[]
with open(args.summary, newline='') as f:
    reader=csv.DictReader(f, delimiter='\t')
    for r in reader:
        try:
            wall=float(r.get('wall_s','nan'))
            rpg=int(r.get('ranks_per_gpu','0'))
            exit_code=int(r.get('exit','1'))
        except Exception:
            continue
        if math.isfinite(wall) and wall>0 and exit_code==0:
            rows.append((wall,rpg,r))
if not rows:
    print('No successful cases found.')
    sys.exit(0)
rows.sort()
best=rows[0]
print('Stage 3.9 recommendation')
print(f'  best_ranks_per_gpu={best[1]}')
print(f'  best_wall_s={best[0]:.6g}')
print(f'  device_counts={best[2].get("device_counts","NA")}')
print('  recommended_env:')
print('    GENESIS_CUDA_DEVICE_POLICY=local_rank')
print(f'    GENESIS_CUDA_MAX_RANKS_PER_DEVICE={best[1]}')
print(f'    GENESIS_CUDA_WORKER_RANKS_PER_DEVICE={best[1]}')
print('  note: this is a launch-level worker aggregation recommendation; keep validating on each production node class.')
