#!/usr/bin/env python3
"""Reproduce the InferenceX chart axes from a kimi-k26-repro run.

    python3 scripts/inferencex_repro_plot.py [results-root] [--tco 1.5]

InferenceX headline chart (kimi-k26, 8K/1K, FP4) plots, per concurrency point:

    x = Interactivity           = tok/s/user = 1000 / ITL_median_ms
    y = Total Tokens per $1 TCO = tok/s/chip * 3600 / (TCO $/chip/hr)

MI355X TCO is $1.50/chip/hr on the page (SemiAnalysis July 2026 survey);
override with --tco to match a different cost tier. tok/s/chip is the run's
output-token throughput divided by GPUs (chips).

Reads every 8K/1K pass result under results-root (default the kimi-k26-repro
shard in the local mirror), prints an InferenceX-shaped table, writes
inferencex_repro.csv, and (if plotly is present) inferencex_repro.html with
tok/$ vs interactivity. Metric keys are probed, not fixed, since guidellm's
report schema drifts; any pass with no readable report is listed loudly.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

TCO_PER_CHIP_HR = 1.5  # MI355X, InferenceX page
args = [a for a in sys.argv[1:]]
if "--tco" in args:
    i = args.index("--tco")
    TCO_PER_CHIP_HR = float(args[i + 1])
    del args[i : i + 2]
ROOT = Path(args[0] if args else "./mi355x-results/results/kimi-k26-repro")


def _walk(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, path + (str(k).lower(),))
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item, path)
    else:
        yield path, obj


def _stat(report, needles, stat, reject=()):
    want = {"p50": ("p50", "median"), "mean": ("mean",)}.get(stat, (stat,))
    best = None
    for keypath, value in _walk(report):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        joined = "/".join(keypath)
        if any(n not in joined for n in needles) or any(r in joined for r in reject):
            continue
        if not any(w in joined for w in want):
            continue
        if best is None or len(keypath) > best[0]:
            best = (len(keypath), float(value))
    return None if best is None else best[1]


def load_report(result, root):
    path = result.get("guidellm_output_path")
    if not path:
        return None
    cands = [Path(path)]
    if "/results/" in path:
        cands.append(root.parent.parent / "results" / path.split("/results/", 1)[1])
    for c in cands:
        if c.is_file():
            try:
                return json.loads(c.read_text())
            except (OSError, json.JSONDecodeError):
                return None
    return None


rows, missing = [], []
for path in sorted(ROOT.glob("experiments/*/*.json")):
    if path.name.endswith("_guidellm.json"):
        continue
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    meta = result.get("metadata") or {}
    if "concurrency" not in meta:
        continue
    if not result.get("success"):
        missing.append((result.get("name", path.name), "run failed"))
        continue
    report = load_report(result, ROOT)
    if report is None:
        missing.append((result.get("name", path.name), "report not found"))
        continue
    itl = _stat(report, ("inter_token_latency",), "p50")
    thru = _stat(report, ("output", "token", "per_second"), "mean") or _stat(
        report, ("output", "token", "per_second"), "p50"
    )
    if not itl or not thru:
        missing.append((result.get("name", path.name), "no itl/throughput in report"))
        continue
    gpus = result.get("gpu_count") or 8
    # InferenceX Tput/Chip is TOTAL tokens (input+output) per chip. Scale the
    # reliably-extracted output tok/s to total via (isl+osl)/osl.
    isl, osl = meta.get("isl"), meta.get("osl")
    total_tps = thru * (isl + osl) / osl if (isl and osl) else thru
    tok_s_chip = total_tps / gpus
    rows.append(
        {
            "concurrency": meta["concurrency"],
            "interactivity_tok_s_user": round(1000.0 / itl, 1),
            "tok_s_chip": round(tok_s_chip, 1),
            "tok_per_dollar_tco": round(tok_s_chip * 3600.0 / TCO_PER_CHIP_HR, 0),
            "ttft_p50_ms": round(_stat(report, ("time_to_first_token",), "p50") or 0, 1),
            "e2e_p50_ms": round(_stat(report, ("request_latency",), "p50", reject=("token",)) or 0, 1),
        }
    )

rows.sort(key=lambda r: r["concurrency"])
if rows:
    out_csv = ROOT / "inferencex_repro.csv"
    with out_csv.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"MI355X vLLM · Kimi-K2.6 8K/1K · TCO ${TCO_PER_CHIP_HR}/chip/hr")
    print(f"{'conc':>5} {'interactivity':>14} {'tok/s/chip':>11} {'tok/$ TCO':>14} {'TTFT ms':>9} {'E2E ms':>9}")
    for r in rows:
        print(
            f"{r['concurrency']:>5} {r['interactivity_tok_s_user']:>14.1f} "
            f"{r['tok_s_chip']:>11.1f} {r['tok_per_dollar_tco']:>14,.0f} "
            f"{r['ttft_p50_ms']:>9.1f} {r['e2e_p50_ms']:>9.1f}"
        )
    print(f"\n-> {out_csv}")
else:
    print("No 8K/1K points with metrics found under", ROOT)

if missing:
    print("\npasses with no usable metrics:")
    for name, why in missing:
        print(f"  {name}: {why}")

try:
    import plotly.graph_objects as go

    if rows:
        fig = go.Figure(
            go.Scatter(
                x=[r["interactivity_tok_s_user"] for r in rows],
                y=[r["tok_per_dollar_tco"] for r in rows],
                mode="lines+markers+text",
                text=[f"c{r['concurrency']}" for r in rows],
                textposition="top center",
                name="MI355X (vLLM) — our repro",
            )
        )
        fig.update_layout(
            title="Kimi K2.6-Code 1T · 8K/1K · FP4 — Total Tokens per $1 TCO vs Interactivity",
            xaxis_title="Interactivity (tok/s/user)",
            yaxis_title=f"Total Tokens per $1 TCO (tok/$, MI355X @ ${TCO_PER_CHIP_HR}/chip/hr)",
        )
        out_html = ROOT / "inferencex_repro.html"
        fig.write_html(str(out_html), include_plotlyjs="cdn")
        print(f"-> {out_html}")
except ImportError:
    pass
