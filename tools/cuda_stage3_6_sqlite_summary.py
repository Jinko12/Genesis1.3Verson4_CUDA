#!/usr/bin/env python3
import sqlite3
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} profile.sqlite", file=sys.stderr)
    sys.exit(2)

path = Path(sys.argv[1])
con = sqlite3.connect(str(path))
con.row_factory = sqlite3.Row

def tables():
    return {r[0] for r in con.execute("select name from sqlite_master where type='table'")}

def columns(table):
    return [r[1] for r in con.execute(f"pragma table_info({table})")]

names = tables()
print(f"SQLite: {path}")
print()

# Nsight schemas vary by version.  Use guarded queries and print what is available.
if "CUPTI_ACTIVITY_KIND_MEMCPY" in names:
    cols = columns("CUPTI_ACTIVITY_KIND_MEMCPY")
    bytes_col = "bytes" if "bytes" in cols else "copySize" if "copySize" in cols else None
    kind_col = "copyKind" if "copyKind" in cols else "copyKindString" if "copyKindString" in cols else None
    if bytes_col:
        print("Memcpy totals:")
        group = kind_col if kind_col else "1"
        for r in con.execute(f"select {group} as kind, count(*) as cnt, sum({bytes_col})/1e6 as MB, max({bytes_col})/1e6 as maxMB from CUPTI_ACTIVITY_KIND_MEMCPY group by {group} order by MB desc"):
            print(f"  {r['kind']}: cnt={r['cnt']} MB={r['MB']:.3f} maxMB={r['maxMB']:.3f}")
        print()

if "CUPTI_ACTIVITY_KIND_KERNEL" in names:
    cols = columns("CUPTI_ACTIVITY_KIND_KERNEL")
    name_col = "demangledName" if "demangledName" in cols else "shortName" if "shortName" in cols else "name" if "name" in cols else None
    dur_expr = "(end-start)/1e9" if "start" in cols and "end" in cols else None
    if name_col and dur_expr:
        print("Top kernels:")
        q = f"select {name_col} as name, count(*) as cnt, sum({dur_expr}) as seconds from CUPTI_ACTIVITY_KIND_KERNEL group by {name_col} order by seconds desc limit 20"
        for r in con.execute(q):
            print(f"  {r['seconds']:.6f}s cnt={r['cnt']} {r['name']}")
        print()

# NVTX events are especially useful for Genesis critical-path ranges.
for table in ["NVTX_EVENTS", "NVTX_EVENTS_NAMED"]:
    if table in names:
        print(f"NVTX table present: {table}")
        cols = columns(table)
        print("  columns:", ", ".join(cols))
        print()
        break
