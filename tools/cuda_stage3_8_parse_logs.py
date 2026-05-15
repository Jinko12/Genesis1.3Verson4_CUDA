#!/usr/bin/env python3
import argparse
import pathlib
import re
from collections import Counter

WALL_RE = re.compile(r"Total Wall Clock Time:\s*([0-9.]+)\s*seconds")
ELAPSED_RE = re.compile(r"ELAPSED=([^\s]+)")
DEVICE_RE = re.compile(r"-> device\s+(\d+)\s+\(")
EXIT_RE = re.compile(r"EXIT=(\d+)")


def parse_log(path: pathlib.Path):
    text = path.read_text(errors="replace")
    wall = None
    for m in WALL_RE.finditer(text):
        wall = float(m.group(1))
    elapsed = None
    for m in ELAPSED_RE.finditer(text):
        elapsed = m.group(1)
    exit_code = None
    for m in EXIT_RE.finditer(text):
        exit_code = int(m.group(1))
    counts = Counter(int(m.group(1)) for m in DEVICE_RE.finditer(text))
    return wall, elapsed, exit_code, counts


def main():
    ap = argparse.ArgumentParser(description="Summarize Stage 3.8 rank/GPU sweep logs")
    ap.add_argument("paths", nargs="+", help="run.log files or directories containing run.log")
    args = ap.parse_args()

    logs = []
    for raw in args.paths:
        p = pathlib.Path(raw)
        if p.is_dir():
            logs.extend(sorted(p.rglob("run.log")))
        elif p.is_file():
            logs.append(p)

    print("case\twall_s\telapsed\texit\tdevice_counts")
    for log in logs:
        wall, elapsed, exit_code, counts = parse_log(log)
        device_counts = ",".join(f"GPU{k}:{v}" for k, v in sorted(counts.items())) or "NA"
        print(f"{log.parent}\t{wall if wall is not None else 'NA'}\t{elapsed or 'NA'}\t{exit_code if exit_code is not None else 'NA'}\t{device_counts}")

if __name__ == "__main__":
    main()
