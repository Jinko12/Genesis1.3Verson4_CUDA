#!/usr/bin/env python3
import argparse, math, sys
from pathlib import Path

try:
    import h5py
    import numpy as np
except Exception as exc:
    print(f"h5py/numpy unavailable: {exc}", file=sys.stderr)
    sys.exit(2)


def compare_dataset(a, b, path, atol, rtol, out):
    if a.shape != b.shape:
        out.append((path, "shape", a.shape, b.shape, None, None))
        return
    if a.dtype.kind in "iufc" and b.dtype.kind in "iufc":
        aa = a[()]
        bb = b[()]
        diff = np.abs(aa - bb)
        max_abs = float(np.max(diff)) if diff.size else 0.0
        denom = np.maximum(np.abs(aa), np.abs(bb))
        rel = np.where(denom > 0, diff / denom, diff)
        max_rel = float(np.max(rel)) if rel.size else 0.0
        ok = np.allclose(aa, bb, atol=atol, rtol=rtol, equal_nan=True)
        if not ok:
            out.append((path, "value", str(a.shape), str(a.dtype), max_abs, max_rel))
    else:
        aa = a[()]
        bb = b[()]
        if aa != bb:
            out.append((path, "value", str(a.shape), str(a.dtype), None, None))


def walk(fa, fb, prefix, atol, rtol, out):
    keys = set(fa.keys()) | set(fb.keys())
    for k in sorted(keys):
        path = f"{prefix}/{k}" if prefix else k
        if k not in fa or k not in fb:
            out.append((path, "missing", k in fa, k in fb, None, None))
            continue
        oa, ob = fa[k], fb[k]
        if isinstance(oa, h5py.Dataset) and isinstance(ob, h5py.Dataset):
            compare_dataset(oa, ob, path, atol, rtol, out)
        elif isinstance(oa, h5py.Group) and isinstance(ob, h5py.Group):
            walk(oa, ob, path, atol, rtol, out)
        else:
            out.append((path, "type", type(oa).__name__, type(ob).__name__, None, None))


def main():
    ap = argparse.ArgumentParser(description="Stage 3.9B physical-correctness HDF5 guard")
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    args = ap.parse_args()
    diffs = []
    with h5py.File(args.reference, "r") as fa, h5py.File(args.candidate, "r") as fb:
        walk(fa, fb, "", args.atol, args.rtol, diffs)
    if diffs:
        print(f"status=FAIL mismatched={len(diffs)}")
        for d in diffs[:50]:
            print("\t".join(str(x) for x in d))
        sys.exit(1)
    print(f"status=OK reference={args.reference} candidate={args.candidate} atol={args.atol} rtol={args.rtol}")

if __name__ == "__main__":
    main()
