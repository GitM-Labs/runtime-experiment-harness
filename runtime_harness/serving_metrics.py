"""Serving metrics in SemiAnalysis InferenceX terms, printed at end of a run.

InferenceX characterises an online-serving point by two axes and the latencies
behind them:

* **tok/s/GPU** -- total output-token throughput divided by GPUs. The
  cost/throughput axis.
* **tok/s/user** -- per-request output speed a single user sees, i.e.
  ``1000 / ITL_median`` (ms). The interactivity axis.
* **TTFT** -- time to first token (prefill responsiveness).
* **ITL** -- inter-token latency (decode smoothness); TPOT is its mean.
* **E2E** -- end-to-end request latency.

Latencies are reported at p50/p90/p99 because a serving SLA lives in the tail,
not the mean. Everything is read from guidellm's own JSON report; guidellm's
schema drifts between releases, so each field is located by probing key paths
rather than a fixed dotted path, and a value that cannot be found prints "-"
rather than a zero that would read as a real measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def _walk(obj, path=()):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _walk(value, path + (str(key).lower(),))
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item, path)
    else:
        yield path, obj


# stat word -> the report key fragments that mean it (median is p50's alias)
_STAT_ALIASES = {
    "p50": ("p50", "median"),
    "p90": ("p90",),
    "p99": ("p99",),
    "mean": ("mean",),
}


def _stat(report: dict, needles: tuple[str, ...], stat: str, reject: tuple[str, ...] = ()):
    """Best numeric value whose key path contains every needle and the stat word.

    Ties break toward the deepest (most specific) path, so a nested
    ``metrics/time_to_first_token_ms/p99`` beats a summary field that happens to
    share a word. Returns None when nothing matches -- never a fabricated 0.
    """
    want = _STAT_ALIASES.get(stat, (stat,))
    best = None
    for keypath, value in _walk(report):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        joined = "/".join(keypath)
        if any(n not in joined for n in needles):
            continue
        if any(r in joined for r in reject):
            continue
        if not any(w in joined for w in want):
            continue
        depth = len(keypath)
        if best is None or depth > best[0]:
            best = (depth, float(value))
    return None if best is None else round(best[1], 2)


def _scalar(report: dict, needles: tuple[str, ...], reject: tuple[str, ...] = ()):
    """Shallowest numeric whose key path contains every needle, ignoring stat
    words. For fields guidellm emits as a bare scalar rather than a stat block."""
    best = None
    for keypath, value in _walk(report):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        joined = "/".join(keypath)
        if any(n not in joined for n in needles) or any(r in joined for r in reject):
            continue
        if best is None or len(keypath) < best[0]:
            best = (len(keypath), float(value))
    return None if best is None else round(best[1], 2)


_TTFT = ("time_to_first_token",)
_ITL = ("inter_token_latency",)
_E2E = ("request_latency",)
_THRU = ("output", "token", "per_second")


def _load_report(result: dict):
    """The guidellm JSON report a result points at, or None.

    Tries the recorded path, then the same path remapped into a local mirror
    (the off-cluster sync keeps the on-cluster ``/…/results/…`` layout under a
    different root), so this works both on the node and against a synced copy.
    """
    path = result.get("guidellm_output_path")
    if not path:
        return None
    candidates = [Path(path)]
    if "/results/" in path:
        tail = path.split("/results/", 1)[1]
        candidates.append(Path.cwd() / "mi355x-results" / "results" / tail)
    for cand in candidates:
        if cand.is_file():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
    return None


def serving_metric_rows(results: list[dict]) -> list[dict]:
    """One metrics row per successful guidellm result (per pass, when multi-pass)."""
    rows = []
    for result in results:
        if result.get("engine") != "guidellm" or not result.get("success"):
            continue
        report = _load_report(result)
        if report is None:
            continue
        itl_p50 = _stat(report, _ITL, "p50")
        itl_p90 = _stat(report, _ITL, "p90")
        out_tps = (
            _stat(report, _THRU, "mean")
            or _stat(report, _THRU, "p50")
            or _scalar(report, _THRU)
        )
        gpus = result.get("gpu_count") or 1
        meta = result.get("metadata") or {}
        isl, osl = meta.get("isl"), meta.get("osl")
        # InferenceX "Tput/Chip" is TOTAL (input+output) tokens/s per chip.
        # Scale the reliably-extracted output tok/s by (isl+osl)/osl.
        total_per_chip = None
        if out_tps is not None and isl and osl:
            total_per_chip = out_tps * (isl + osl) / osl / gpus
        ttft_p50 = _stat(report, _TTFT, "p50")
        ttft_p90 = _stat(report, _TTFT, "p90")
        row = {
            "name": result.get("name", "?"),
            "concurrency": meta.get("concurrency"),
            "chips": gpus,
            "precision": result.get("precision", "?"),
            "parallelism": result.get("parallelism", f"TP{gpus}"),
            "total_tok_s_chip": round(total_per_chip, 1) if total_per_chip else None,
            # Interactivity = 1000 / ITL. P90 interactivity uses ITL_p90.
            "int_p50": round(1000.0 / itl_p50, 2) if itl_p50 else None,
            "int_p90": round(1000.0 / itl_p90, 2) if itl_p90 else None,
            # TTFT in seconds, matching InferenceX.
            "ttft_p50_s": round(ttft_p50 / 1000.0, 3) if ttft_p50 else None,
            "ttft_p90_s": round(ttft_p90 / 1000.0, 3) if ttft_p90 else None,
            "e2e_p50_s": _stat(report, _E2E, "p50", reject=("token",)),
        }
        if any(row[k] is not None for k in ("total_tok_s_chip", "int_p50", "ttft_p50_s")):
            rows.append(row)
    return rows


def failure_rows(results: list[dict]) -> list[dict]:
    """Failed guidellm runs with a concrete reason and the log to read."""
    frows = []
    for result in results:
        if result.get("engine") != "guidellm" or result.get("success"):
            continue
        if result.get("status") == "skipped":
            continue
        frows.append(
            {
                "name": result.get("name", "?"),
                "parallelism": result.get("parallelism", "?"),
                "reason": result.get("failure_reason")
                or result.get("stderr", "").strip().splitlines()[-1:] or "unknown",
                "server_log": result.get("server_log"),
            }
        )
    return frows


def _fmt(value):
    return "-" if value is None else f"{value:,.1f}"


def render_serving_metrics(rows: list[dict]) -> Table:
    """InferenceX-format results table (Concurrency / Chips / Precision /
    Parallelism / Tput per chip / Interactivity / TTFT)."""
    table = Table(title="results (SemiAnalysis InferenceX format)")
    table.add_column("experiment", overflow="fold")
    table.add_column("Conc", justify="right")
    table.add_column("Chips", justify="right")
    table.add_column("Prec")
    table.add_column("Parallel")
    table.add_column("Tput/Chip", justify="right")
    table.add_column("P50 Int", justify="right")
    table.add_column("P90 Int", justify="right")
    table.add_column("P50 TTFT s", justify="right")
    table.add_column("P90 TTFT s", justify="right")
    for row in sorted(rows, key=lambda r: (r["name"], r["concurrency"] or 0)):
        table.add_row(
            row["name"],
            "-" if row["concurrency"] is None else str(row["concurrency"]),
            str(row["chips"]),
            str(row["precision"]),
            str(row["parallelism"]),
            "-" if row["total_tok_s_chip"] is None else f"{row['total_tok_s_chip']:,.1f}",
            _fmt(row["int_p50"]),
            _fmt(row["int_p90"]),
            "-" if row["ttft_p50_s"] is None else f"{row['ttft_p50_s']:.3f}",
            "-" if row["ttft_p90_s"] is None else f"{row['ttft_p90_s']:.3f}",
        )
    return table


def render_failures(frows: list[dict]) -> Table:
    table = Table(title="FAILURES — concrete reason per failed run")
    table.add_column("experiment", overflow="fold")
    table.add_column("Parallel")
    table.add_column("reason", overflow="fold")
    table.add_column("log", overflow="fold")
    for f in frows:
        reason = f["reason"]
        if isinstance(reason, list):
            reason = reason[0] if reason else "unknown"
        table.add_row(f["name"], str(f["parallelism"]), str(reason), f.get("server_log") or "-")
    return table


def print_serving_metrics(results: list[dict]) -> list[dict]:
    """On a finished run: print the InferenceX-format results table for the
    successful passes, and a FAILURES table (with a concrete reason and the log
    path) for any that failed. Returns the successful rows.
    """
    rows = serving_metric_rows(results)
    if rows:
        console.print(render_serving_metrics(rows))
    frows = failure_rows(results)
    if frows:
        console.print(render_failures(frows))
    if not rows and not frows:
        console.log("[dim]no serving results to summarize (reports not found)[/dim]")
    return rows
