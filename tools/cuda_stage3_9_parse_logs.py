#!/usr/bin/env python3
import argparse
import pathlib
import re
from collections import Counter

WALL_RE = re.compile(r"Total Wall Clock Time:\s*([0-9.]+)\s*seconds")
ELAPSED_RE = re.compile(r"ELAPSED=([^\s]+)")
EXIT_RE = re.compile(r"EXIT=(\d+)")
MAP_RE = re.compile(r"-> device\s+(\d+)\s+\(")
SUMMARY_RE = re.compile(r"GENESIS CUDA per-GPU worker summary:.*?(GPU\d+=\d+(?:,\s*GPU\d+=\d+)*)")
RPG_RE = re.compile(r"ranks_per_gpu=(\d+)")
NP_RE = re.compile(r"mpi_ranks=(\d+)")

def parse(path: pathlib.Path):
    text = path.read_text(errors='replace')
    wall = None
    for m in WALL_RE.finditer(text):
        wall = float(m.group(1))
    elapsed = None
    for m in ELAPSED_RE.finditer(text):
        elapsed = m.group(1)
    exit_code = None
    for m in EXIT_RE.finditer(text):
        exit_code = int(m.group(1))
    rpg = None
    for m in RPG_RE.finditer(text):
        rpg = int(m.group(1))
    np = None
    for m in NP_RE.finditer(text):
        np = int(m.group(1))
    summary = None
    for m in SUMMARY_RE.finditer(text):
        summary = m.group(1).replace(' ', '')
    if summary is None:
        counts = Counter(int(m.group(1)) for m in MAP_RE.finditer(text))
        if counts:
            summary = ','.join(f'GPU{k}={v}' for k, v in sorted(counts.items()))
    return wall, elapsed, exit_code, rpg, np, summary or 'NA'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    args = ap.parse_args()
    logs = []
    for raw in args.paths:
        p = pathlib.Path(raw)
        if p.is_dir():
            logs.extend(sorted(p.rglob('run.log')))
        elif p.is_file():
            logs.append(p)
    print('case\tranks_per_gpu\tmpi_ranks\twall_s\telapsed\texit\tdevice_counts')
    for log in logs:
        wall, elapsed, exit_code, rpg, np, summary = parse(log)
        print(f'{log.parent}\t{rpg if rpg is not None else "NA"}\t{np if np is not None else "NA"}\t{wall if wall is not None else "NA"}\t{elapsed or "NA"}\t{exit_code if exit_code is not None else "NA"}\t{summary}')

if __name__ == '__main__':
    main()
