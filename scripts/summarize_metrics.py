#!/usr/bin/env python3
"""Print InferenceX-format results (+ a concrete-reason FAILURES table) for a
synced results tree, using the SAME renderer the harness prints at end-of-run.

    python3 scripts/summarize_metrics.py [results-root]

results-root defaults to ./mi355x-results/results. Loads every run's result
JSON and delegates to runtime_harness.serving_metrics, so the local summary is
byte-for-byte the table you saw in the pod log:

    experiment | Conc | Chips | Prec | Parallel | Tput/Chip (total tok/s/chip)
               | P50 Int | P90 Int | P50 TTFT s | P90 TTFT s
    FAILURES   | experiment | Parallel | reason | log

Also writes all_metrics.csv (successful rows) next to the root.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from runtime_harness import serving_metrics  # noqa: E402

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "./mi355x-results/results")

results = []
for path in sorted(ROOT.glob("*/experiments/*/*.json")):
    if path.name.endswith(("_guidellm.json", "_telemetry.json")):
        continue
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    if isinstance(result, dict) and result.get("engine") in ("guidellm", None):
        # Tag the shard so names are unambiguous across shards.
        result.setdefault("shard", path.relative_to(ROOT).parts[0])
        results.append(result)

if not results:
    print(f"No result JSONs found under {ROOT}. Has sync_results.sh pulled them yet?")
    sys.exit(0)

rows = serving_metrics.print_serving_metrics(results)

if rows:
    out_csv = ROOT / "all_metrics.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} successful rows -> {out_csv}")
