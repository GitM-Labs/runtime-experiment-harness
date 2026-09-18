#!/usr/bin/env python3
"""Summarize where GPU time goes, from a rocprofv3 kernel-trace CSV.

    python3 scripts/rocprof_kernel_summary.py <trace_kernel_trace.csv> [--top 25]

Answers the load-bearing question for the InferenceX throughput gap: is the
MoE expert GEMM the bottleneck (and how dominant), or is time going to
attention / collectives / dequant / elementwise? Aggregates total kernel
duration by kernel name, classifies each into a coarse bucket, and prints the
bucket shares plus the heaviest individual kernels.

rocprofv3 kernel CSV columns vary by version; this finds the name and the
start/end (or duration) columns by header, so it works across builds. Point it
at any run's telemetry/rocprof/trace_kernel_trace.csv in the local mirror.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

args = sys.argv[1:]
top_n = 25
if "--top" in args:
    i = args.index("--top")
    top_n = int(args[i + 1])
    del args[i : i + 2]
if not args:
    sys.exit("usage: rocprof_kernel_summary.py <trace_kernel_trace.csv> [--top N]")
path = Path(args[0])

# Coarse buckets by kernel-name regex, first match wins. Order matters:
# MoE before generic GEMM, since expert GEMMs are also matmuls.
BUCKETS = [
    ("moe/expert", re.compile(r"moe|expert|grouped_?gemm|fused_moe|topk|routing|gate", re.I)),
    ("attention/MLA", re.compile(r"attn|attention|mla|flash|paged|rope|softmax", re.I)),
    ("gemm/linear", re.compile(r"gemm|matmul|cutlass|wvsplitk|hipblas|linear|dense", re.I)),
    ("quant/dequant", re.compile(r"dequant|quant|w4a16|w4a4|fp4|int4|scale|convert|cast", re.I)),
    ("collective", re.compile(r"rccl|nccl|all_?reduce|all_?gather|reduce_?scatter|broadcast|cross_device_reduce|device_reduce|custom_?ar", re.I)),
    ("norm/elementwise", re.compile(r"norm|rms|silu|gelu|add|mul|elementwise|activation|residual", re.I)),
    ("copy/memset", re.compile(r"memcpy|memset|copy|cast_|fill", re.I)),
]


def bucket_of(name: str) -> str:
    for label, rx in BUCKETS:
        if rx.search(name):
            return label
    return "other"


def find_col(header, *cands):
    low = [h.lower() for h in header]
    for cand in cands:
        for i, h in enumerate(low):
            if cand in h:
                return i
    return None


with path.open(newline="") as handle:
    reader = csv.reader(handle)
    header = next(reader)
    name_i = find_col(header, "kernel_name", "name")
    dur_i = find_col(header, "duration")
    start_i = find_col(header, "start_timestamp", "start")
    end_i = find_col(header, "end_timestamp", "end")
    if name_i is None or (dur_i is None and (start_i is None or end_i is None)):
        sys.exit(f"could not find name/duration columns in header: {header}")

    by_name = defaultdict(lambda: [0, 0.0])  # name -> [count, total_ns]
    total = 0.0
    for row in reader:
        if not row or len(row) <= name_i:
            continue
        try:
            dur = float(row[dur_i]) if dur_i is not None else float(row[end_i]) - float(row[start_i])
        except (ValueError, IndexError):
            continue
        if dur < 0:
            continue
        name = row[name_i]
        by_name[name][0] += 1
        by_name[name][1] += dur
        total += dur

if total <= 0:
    sys.exit("no positive kernel durations parsed (truncated trace? wrong column?)")

buckets = defaultdict(lambda: [0, 0.0])
for name, (count, dur) in by_name.items():
    b = bucket_of(name)
    buckets[b][0] += count
    buckets[b][1] += dur

print(f"kernel trace: {path}")
print(f"total kernel time: {total / 1e9:.3f} s across {sum(c for c, _ in by_name.values()):,} launches\n")

print(f"{'bucket':<20} {'time %':>8} {'time (s)':>12} {'launches':>12}")
for b, (count, dur) in sorted(buckets.items(), key=lambda kv: -kv[1][1]):
    print(f"{b:<20} {100 * dur / total:>7.1f}% {dur / 1e9:>12.3f} {count:>12,}")

print(f"\ntop {top_n} kernels by total time:")
print(f"{'time %':>7} {'time (s)':>11} {'launches':>11}  kernel")
for name, (count, dur) in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:top_n]:
    short = name if len(name) <= 90 else name[:87] + "..."
    print(f"{100 * dur / total:>6.1f}% {dur / 1e9:>11.3f} {count:>11,}  {short}")
