#!/usr/bin/env python3
import re
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: cuda_stage3_8_parse_profile_summary.py run.log [run.log ...]", file=sys.stderr)
    sys.exit(2)

for arg in sys.argv[1:]:
    p = Path(arg)
    txt = p.read_text(errors="replace")
    wall = None
    m = re.findall(r"Total Wall Clock Time:\s*([0-9.]+)\s*seconds", txt)
    if m:
        wall = float(m[-1])
    elapsed = None
    m = re.findall(r"ELAPSED=([^\s]+)", txt)
    if m:
        elapsed = m[-1]
    exit_code = None
    m = re.findall(r"EXIT=([^\s]+)", txt)
    if m:
        exit_code = m[-1]
    gpu_counts = {}
    for mm in re.finditer(r"-> device\s+(\d+)\s+\(", txt):
        dev = int(mm.group(1))
        gpu_counts[dev] = gpu_counts.get(dev, 0) + 1
    print(f"{p}\twall={wall}\telapsed={elapsed}\texit={exit_code}\tgpu_counts={gpu_counts}")
