#!/usr/bin/env python3
import argparse, os, re, sys
from pathlib import Path

AUDIT_RE = re.compile(r"GENESIS CUDA memory audit: rank=(?P<rank>\d+)/(?P<size>\d+) device=(?P<device>-?\d+) currentMiB=(?P<current>[0-9.]+) peakMiB=(?P<peak>[0-9.]+).*cufftWorkspaceEstimatePeakMiB=(?P<cufft>[0-9.]+)")
WALL_RE = re.compile(r"Total Wall Clock Time:\s*([0-9.]+)\s*seconds")
RPG_RE = re.compile(r"ranks_per_gpu=(\d+) total_ranks=(\d+)")


def parse_log(path: Path):
    text = path.read_text(errors="replace")
    m = RPG_RE.search(text)
    rpg = int(m.group(1)) if m else None
    total = int(m.group(2)) if m else None
    wall = None
    for wm in WALL_RE.finditer(text):
        wall = float(wm.group(1))
    audits = []
    for am in AUDIT_RE.finditer(text):
        audits.append({
            "rank": int(am.group("rank")),
            "size": int(am.group("size")),
            "device": int(am.group("device")),
            "current": float(am.group("current")),
            "peak": float(am.group("peak")),
            "cufft": float(am.group("cufft")),
        })
    peak_max = max((a["peak"] for a in audits), default=0.0)
    peak_sum = sum(a["peak"] for a in audits)
    cufft_sum = sum(a["cufft"] for a in audits)
    return {"path": path, "rpg": rpg, "total": total, "wall": wall, "audits": audits, "peak_max": peak_max, "peak_sum": peak_sum, "cufft_sum": cufft_sum}


def iter_logs(p: Path):
    if p.is_file():
        yield p
    else:
        for log in sorted(p.rglob("run.log")):
            yield log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tsv", action="store_true")
    ap.add_argument("--recommend", action="store_true")
    args = ap.parse_args()
    results = [parse_log(p) for p in iter_logs(Path(args.path))]
    if args.tsv:
        for r in results:
            exit_code = 0 if r["wall"] is not None else 1
            print(f"{r['rpg'] or ''}\t{r['total'] or ''}\t{exit_code}\t{r['wall'] or ''}\t{r['peak_max']:.3f}\t{r['peak_sum']:.3f}\t{r['cufft_sum']:.3f}\t{r['path']}")
        return
    if args.recommend:
        valid = [r for r in results if r["wall"] is not None]
        if not valid:
            print("No successful Stage 3.9B memory-audit runs found.")
            return
        best = min(valid, key=lambda r: r["wall"])
        lowest_mem = min(valid, key=lambda r: r["peak_sum"])
        print("Stage 3.9B memory audit recommendation")
        print("======================================")
        print(f"Best wall time: ranks_per_gpu={best['rpg']} total_ranks={best['total']} wall_s={best['wall']:.6g} peak_sum_mib={best['peak_sum']:.3f}")
        print(f"Lowest summed GPU peak memory: ranks_per_gpu={lowest_mem['rpg']} total_ranks={lowest_mem['total']} peak_sum_mib={lowest_mem['peak_sum']:.3f} wall_s={lowest_mem['wall']:.6g}")
        print("Recommended next check: compare peak categories in each run.log and prioritize duplicated field/source/cuFFT/diagnostics buffers.")
        return
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
