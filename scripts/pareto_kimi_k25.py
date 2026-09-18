#!/usr/bin/env python3
"""Build InferenceX-style Pareto curves from the kimi-k25-ix-pareto sweep.

    python3 scripts/pareto_kimi_k25.py [results-root]

results-root defaults to ./mi355x-results/results/kimi-k25-ix-pareto.
Reads every harness result JSON under experiments/ (one per (shape, conc)
pass; each carries metadata {isl, osl, concurrency} and a pointer to its
guidellm report), extracts throughput and per-user speed, writes
pareto_points.csv and pareto.html (plotly, if installed) next to the root.

Axes per shape:
  x = tok/s/user  (per-request output speed; median across requests)
  y = tok/s/GPU   (total output token throughput / gpu_count)
The Pareto frontier is the subset not dominated in both axes.

guidellm report schemas drift between versions, so metric extraction probes
several plausible key paths and REFUSES silently-empty output: any pass whose
report yields no metrics is listed loudly at the end.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def _walk(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, path + (str(k).lower(),))
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item, path)
    else:
        yield path, obj


def _find_metric(report: dict, *needles, reject=("request",)):
    """Best numeric value whose key path contains every needle (median > mean)."""
    best = None
    for path, value in _walk(report):
        if not isinstance(value, (int, float)):
            continue
        joined = "/".join(path)
        if any(n not in joined for n in needles):
            continue
        if any(r in joined for r in reject):
            continue
        rank = 2 if "median" in joined else 1 if "mean" in joined else 0
        if best is None or rank > best[0]:
            best = (rank, float(value))
    return None if best is None else best[1]


def extract_point(result: dict):
    """(tok/s total, tok/s/user) from one harness result + its guidellm report."""
    report_path = result.get("guidellm_output_path")
    if not report_path:
        return None, None
    # The synced mirror keeps the on-cluster layout; retarget /mnt/shared paths.
    candidates = [Path(report_path)]
    if "/results/" in report_path:
        candidates.append(ROOT.parent / report_path.split("/results/", 1)[1])
    report = None
    for cand in candidates:
        if cand.is_file():
            report = json.loads(cand.read_text())
            break
    if report is None:
        return None, None
    total = _find_metric(report, "output", "token", "per_second") or _find_metric(
        report, "output_tokens_per_second"
    )
    per_user = _find_metric(report, "output_tokens_per_second", "per_request") or _find_metric(
        report, "tokens_per_second", "request"
    )
    itl_ms = _find_metric(report, "inter_token_latency")
    if per_user is None and itl_ms:
        per_user = 1000.0 / itl_ms
    lat = {
        "ttft_ms": _find_metric(report, "time_to_first_token"),
        "itl_ms": itl_ms,
        "e2e_ms": _find_metric(report, "request_latency")
        or _find_metric(report, "latency", reject=("token",)),
    }
    return total, per_user, lat


def pareto_front(points):
    """Points (x, y, label) not dominated by any other in both axes."""
    front = []
    for p in points:
        if not any(q[0] >= p[0] and q[1] >= p[1] and q != p for q in points):
            front.append(p)
    return sorted(front)


ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "./mi355x-results/results/kimi-k25-ix-pareto")

rows, missing = [], []
for path in sorted(ROOT.glob("experiments/*/*.json")):
    if path.name.endswith("_guidellm.json"):
        continue
    result = json.loads(path.read_text())
    meta = result.get("metadata") or {}
    if "concurrency" not in meta:
        continue
    total, per_user, _lat = extract_point(result)
    if not result.get("success"):
        missing.append((path.name, "run failed"))
        continue
    if total is None or per_user is None:
        missing.append((path.name, "no metrics extracted - inspect report keys"))
        continue
    gpus = result.get("gpu_count") or 8
    rows.append(
        {
            "shape": f"{meta['isl']}/{meta['osl']}",
            "concurrency": meta["concurrency"],
            "tok_s_total": round(total, 2),
            "tok_s_per_gpu": round(total / gpus, 2),
            "tok_s_per_user": round(per_user, 2),
            "source": path.name,
        }
    )

rows.sort(key=lambda r: (r["shape"], r["concurrency"]))
out_csv = ROOT / "pareto_points.csv"
with out_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["empty"])
    writer.writeheader()
    writer.writerows(rows)
print(f"{len(rows)} points -> {out_csv}")

for shape in sorted({r["shape"] for r in rows}):
    pts = [(r["tok_s_per_user"], r["tok_s_per_gpu"], f"c{r['concurrency']}") for r in rows if r["shape"] == shape]
    front = pareto_front(pts)
    print(f"\n{shape} (ISL/OSL) — frontier ({len(front)}/{len(pts)} points):")
    for x, y, label in front:
        print(f"  {label:>5}: {y:9.1f} tok/s/GPU @ {x:7.1f} tok/s/user")

if missing:
    print("\nWARNING - passes with no usable metrics (fix before publishing):")
    for name, why in missing:
        print(f"  {name}: {why}")

try:
    import plotly.graph_objects as go

    fig = go.Figure()
    for shape in sorted({r["shape"] for r in rows}):
        pts = sorted(
            [(r["tok_s_per_user"], r["tok_s_per_gpu"], r["concurrency"]) for r in rows if r["shape"] == shape]
        )
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in pts],
                y=[p[1] for p in pts],
                mode="lines+markers+text",
                text=[f"c{p[2]}" for p in pts],
                textposition="top center",
                name=shape,
            )
        )
    fig.update_layout(
        title="Kimi K2.5 on 8x MI355X - vLLM lane Pareto (per ISL/OSL shape)",
        xaxis_title="tok/s/user (per-request output speed)",
        yaxis_title="tok/s/GPU (total output throughput / GPUs)",
    )
    out_html = ROOT / "pareto.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"\nplot -> {out_html}")
except ImportError:
    print("\nplotly not installed locally; CSV + frontier printout only.")
