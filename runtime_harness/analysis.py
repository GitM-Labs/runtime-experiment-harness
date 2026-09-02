"""Read what a `gitm capture serve` run left behind and turn it into tables.

Three views, in increasing order of what they need from the capture:

* **buckets** -- kernel time by taxonomy bucket (moe / gemm / elementwise / ...),
  read straight out of ``kernel_breakdown.json``. Every traced arm has it.
* **layerwise** -- kernel time by (layer, op), recovered from the NVTX ranges. Only
  the ``nvtx`` arm has these; every other arm's kernels carry ``range_op: null``.
* **phase** -- prefill vs decode. Needs gitm's own classifier, so it is computed
  in-process when gitm is importable and skipped with a note when it is not.

Everything here reads files. Nothing re-runs a capture, so an analysis can be
pointed at a directory captured hours ago on another box.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# What "the same workload got slower" is measured on. The server block is vLLM's
# own Prometheus histograms differenced across the window; the client block is
# this harness's view. They are reported separately rather than averaged -- when
# they disagree the difference is the client's own scheduling, which is exactly
# what a tracing-overhead measurement must not silently absorb.
THROUGHPUT_KEY = "output_tokens_per_s"


def read_json(path):
    """Parse a JSON artifact, or return None when it is absent or unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_capture_dir(path) -> bool:
    """A directory gitm wrote a capture into.

    run_manifest.json is the marker rather than trace.jsonl: the untraced baseline
    arm has no trace at all, and excluding it here would quietly drop the one arm
    an overhead comparison is measured against.
    """
    path = Path(path)
    return path.is_dir() and any(
        (path / name).is_file()
        for name in ("run_manifest.json", "serving_summary.json", "kernel_breakdown.json")
    )


def discover_capture_dirs(paths):
    """Expand each path into the capture directories under it.

    A path that is itself a capture is taken as-is; otherwise its immediate
    children are scanned. That makes `rex compare experiments/qwen-gpu1` work on the
    directory a sweep wrote, not just on the eight arm directories inside it.
    """
    found = []
    for raw in paths:
        path = Path(raw)
        if is_capture_dir(path):
            found.append(path)
            continue
        if path.is_dir():
            found.extend(sorted(child for child in path.iterdir() if is_capture_dir(child)))
    # Deduplicate while keeping discovery order -- overlapping globs are normal.
    seen = {}
    for path in found:
        seen[path.resolve()] = path
    return list(seen.values())


# --- kernel buckets ---------------------------------------------------------


def bucket_rows(breakdown):
    """Rows of ``kernel_breakdown.json``'s bucket list, largest share first."""
    if not breakdown:
        return []
    rows = []
    for bucket in breakdown.get("buckets", []):
        rows.append(
            {
                "bucket": bucket.get("bucket"),
                "kernels": bucket.get("n_kernels", 0),
                "time_s": bucket.get("time_ns", 0) / 1e9,
                "share": bucket.get("share"),
                "top_names": bucket.get("top_names", [])[:3],
            }
        )
    rows.sort(key=lambda row: row["time_s"], reverse=True)
    return rows


# --- layerwise (NVTX) -------------------------------------------------------


def layerwise_summary(trace_path, top_ops: int = 3):
    """Kernel time per layer and per op, from the NVTX range identity on each kernel.

    Streams the merged trace rather than the per-process shards: the merged file is
    already windowed to the capture, so weight loading, torch.compile and CUDA-graph
    capture -- around 80 seconds of kernels -- are already out of it. Reading the
    shards directly puts all of that back in.

    ``range_op`` with no ``range_layer`` is normal and not an error: ops outside the
    decoder stack (``logits_processor``, ``embed_tokens``) belong to no layer. They
    count toward the per-op table and are excluded from the per-layer one, which is
    why the two totals differ.

    Returns None when the trace has no ranges at all -- an arm captured without
    ``--nvtx``, where every kernel carries ``range_op: null``.
    """
    trace_path = Path(trace_path)
    if not trace_path.is_file():
        return None

    per_layer: dict[int, int] = {}
    per_pair: dict[tuple[int, str], int] = {}
    per_op: dict[str, list] = {}  # op -> [ns, kernels, {layers}]
    n_kernels = n_resolved = 0

    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue  # a killed process leaves a partial trailing line
            if record.get("kind") != "kernel":
                continue
            start, end = record.get("start_ns"), record.get("end_ns")
            if start is None or end is None:
                continue
            n_kernels += 1

            op = record.get("range_op")
            if not op:
                continue
            duration = max(0, end - start)
            layer = record.get("range_layer")

            slot = per_op.setdefault(op, [0, 0, set()])
            slot[0] += duration
            slot[1] += 1
            if layer is not None:
                n_resolved += 1
                slot[2].add(layer)
                per_layer[layer] = per_layer.get(layer, 0) + duration
                per_pair[(layer, op)] = per_pair.get((layer, op), 0) + duration

    if not per_op:
        return None

    layer_total = sum(per_layer.values()) or 1
    op_total = sum(slot[0] for slot in per_op.values()) or 1

    layers = []
    for layer in sorted(per_layer):
        tops = sorted(
            ((ns, op) for (this_layer, op), ns in per_pair.items() if this_layer == layer),
            reverse=True,
        )[:top_ops]
        layers.append(
            {
                "layer": layer,
                "ms": per_layer[layer] / 1e6,
                "share": per_layer[layer] / layer_total,
                "top_ops": [{"op": op, "ms": ns / 1e6} for ns, op in tops],
            }
        )

    ops = [
        {
            "op": op,
            "ms": ns / 1e6,
            "share": ns / op_total,
            "kernels": count,
            "layers": len(seen_layers),
        }
        for op, (ns, count, seen_layers) in sorted(per_op.items(), key=lambda kv: -kv[1][0])
    ]

    return {
        "n_kernels": n_kernels,
        "n_resolved_with_layer": n_resolved,
        "resolved_share": n_resolved / n_kernels if n_kernels else None,
        "per_layer": layers,
        "per_op": ops,
    }


# --- prefill / decode -------------------------------------------------------


def phase_summary(trace_path):
    """Prefill vs decode split, using gitm's own phase classifier.

    Reported with the attribution stats attached, because they govern how much of
    it to believe: only some kernels name their own phase and the rest inherit the
    nearest one that does. A median gap of microseconds means the neighbour is in
    the same engine step; milliseconds means the inference crossed a step boundary,
    and under chunked prefill a single step legitimately mixes both phases.

    Returns ``{"unavailable": reason}`` rather than raising when gitm is not
    importable -- the other two views still stand without it.
    """
    trace_path = Path(trace_path)
    if not trace_path.is_file():
        return None
    try:
        from gitm.optimizer.deviation import stream_by_phase
    except ImportError as exc:
        return {"unavailable": f"gitm is not importable in this environment ({exc})."}

    by_phase, stats = stream_by_phase(trace_path)
    totals = {phase: sum(slot[1] for slot in buckets.values()) for phase, buckets in by_phase.items()}
    grand = sum(totals.values()) or 1

    phases = {}
    for phase, buckets in by_phase.items():
        phases[phase] = {
            "ms": totals[phase] / 1e6,
            "share": totals[phase] / grand,
            "kernels": sum(slot[0] for slot in buckets.values()),
            "buckets": [
                {"bucket": bucket, "ms": ns / 1e6, "kernels": count,
                 "share": ns / (totals[phase] or 1)}
                for bucket, (count, ns) in sorted(buckets.items(), key=lambda kv: -kv[1][1])
            ],
        }

    named = stats["n_direct"] + stats["n_inferred"] + stats["n_unknown"]
    return {
        "phases": phases,
        "attribution": {
            **stats,
            "direct_share": stats["n_direct"] / named if named else None,
            "gap_median_us": stats["gap_median_ns"] / 1e3,
            "gap_max_us": stats["gap_max_ns"] / 1e3,
        },
    }


# --- serving throughput -----------------------------------------------------


def serving_row(summary):
    """The throughput/latency line for one run, from ``serving_summary.json``.

    Throughput comes from the *server* block. The client block is
    ``ServingSummary``, which reports latency percentiles and goodput but carries
    no token counts at all -- there is no ``client.output_tokens_per_s`` to read,
    and asking for one yields None rather than a number.
    """
    if not summary:
        return None
    server = summary.get("server") or {}
    client = summary.get("client") or {}
    return {
        "tracing": summary.get("tracing"),
        "nvtx": summary.get("nvtx"),
        "wall_s": summary.get("wall_s"),
        "output_tokens_per_s": server.get(THROUGHPUT_KEY),
        "tpot_mean_ms": _ms(server.get("tpot_mean_s")),
        "ttft_mean_ms": _ms(server.get("ttft_mean_s")),
        "generation_tokens": server.get("generation_tokens"),
        "client_tpot_p50_ms": _ms(client.get("tpot_p50_s")),
        "client_ttft_p50_ms": _ms(client.get("ttft_p50_s")),
        "n_requests": client.get("n_requests"),
        "n_failed_requests": client.get("n_failed_requests"),
    }


def _ms(seconds):
    return None if seconds is None else seconds * 1e3


# --- one capture directory --------------------------------------------------


def analyze_capture(capture_dir, layerwise: bool = True, phase: bool = True):
    """Every view a single capture directory supports, as one JSON-able record."""
    capture_dir = Path(capture_dir)
    trace_path = capture_dir / "trace.jsonl"
    breakdown = read_json(capture_dir / "kernel_breakdown.json")
    summary = read_json(capture_dir / "serving_summary.json")
    manifest = read_json(capture_dir / "run_manifest.json")

    record = {
        "capture_dir": str(capture_dir),
        "served_model": (manifest or {}).get("served_model"),
        "serve_argv": (manifest or {}).get("serve_argv"),
        "load": (manifest or {}).get("load"),
        "serving": serving_row(summary),
        "n_kernels": (breakdown or {}).get("n_kernels"),
        "kernel_time_s": ((breakdown or {}).get("kernel_time_ns") or 0) / 1e9 or None,
        "gpu_active_share": (breakdown or {}).get("gpu_active_share"),
        "buckets": bucket_rows(breakdown),
        "warnings": (breakdown or {}).get("warnings", []),
    }
    # Three different reasons for an absent layer table -- no trace at all (the
    # untraced arm), a trace with no ranges (any arm but nvtx), and "you asked me
    # not to". They read identically in the output unless recorded separately.
    record["has_trace"] = trace_path.is_file()
    record["layerwise"] = layerwise_summary(trace_path) if layerwise else None
    record["layerwise_skipped"] = not layerwise
    record["phase"] = phase_summary(trace_path) if phase else None
    return record


# --- across arms ------------------------------------------------------------


def overhead_rows(records):
    """Median throughput and TPOT per tracing arm, and the cost against the baseline.

    Grouped on what the run *reported* in ``serving_summary.json["tracing"]``, not
    on the flag the harness passed. Those disagree exactly when the measurement is
    worthless -- an injection variable left over in the environment puts the
    collector into the arm that is supposed to have none -- and then the honest
    reading of "collection is free" is that no baseline was taken.
    """
    grouped: dict[str, list] = {}
    for record in records:
        serving = record.get("serving") or {}
        arm = serving.get("tracing") or record.get("arm") or "unknown"
        grouped.setdefault(arm, []).append(serving)

    rows = []
    for arm, runs in grouped.items():
        throughputs = [run[THROUGHPUT_KEY] for run in runs if run.get(THROUGHPUT_KEY) is not None]
        tpots = [run["tpot_mean_ms"] for run in runs if run.get("tpot_mean_ms") is not None]
        rows.append(
            {
                "arm": arm,
                "runs": len(runs),
                "output_tokens_per_s": statistics.median(throughputs) if throughputs else None,
                "spread": (max(throughputs) - min(throughputs)) if len(throughputs) > 1 else None,
                "tpot_mean_ms": statistics.median(tpots) if tpots else None,
                "n_missing_throughput": len(runs) - len(throughputs),
            }
        )

    order = {"off": 0, "cupti": 1, "cupti+nvtx": 2}
    rows.sort(key=lambda row: (order.get(row["arm"], 99), row["arm"]))

    baseline = next((row for row in rows if row["arm"] == "off"), None)
    base_tps = baseline["output_tokens_per_s"] if baseline else None
    for row in rows:
        throughput = row["output_tokens_per_s"]
        row["throughput_cost"] = (
            (base_tps - throughput) / base_tps
            if base_tps and throughput is not None
            else None
        )
    return rows


# --- rendering --------------------------------------------------------------


def render_buckets(record):
    table = Table(title=f"kernel time by bucket — {Path(record['capture_dir']).name}")
    table.add_column("bucket")
    table.add_column("kernels", justify="right")
    table.add_column("time_s", justify="right")
    table.add_column("share", justify="right")
    for row in record["buckets"]:
        table.add_row(
            str(row["bucket"]),
            f"{row['kernels']:,}",
            f"{row['time_s']:.3f}",
            "" if row["share"] is None else f"{row['share']:.1%}",
        )
    return table


def render_layerwise_ops(layerwise, limit: int = 20):
    table = Table(title="kernel time by op (NVTX ranges)")
    table.add_column("op")
    table.add_column("ms", justify="right")
    table.add_column("share", justify="right")
    table.add_column("kernels", justify="right")
    table.add_column("layers", justify="right")
    for row in layerwise["per_op"][:limit]:
        table.add_row(
            row["op"], f"{row['ms']:.1f}", f"{row['share']:.1%}",
            f"{row['kernels']:,}", str(row["layers"]),
        )
    return table


def render_layers(layerwise, limit: int = 40):
    table = Table(title="kernel time by layer")
    table.add_column("layer", justify="right")
    table.add_column("ms", justify="right")
    table.add_column("share", justify="right")
    table.add_column("top ops")
    for row in layerwise["per_layer"][:limit]:
        tops = " ".join(f"{op['op']}={op['ms']:.1f}" for op in row["top_ops"])
        table.add_row(str(row["layer"]), f"{row['ms']:.1f}", f"{row['share']:.1%}", tops)
    return table


def render_phase(phase):
    table = Table(title="prefill vs decode")
    table.add_column("phase")
    table.add_column("ms", justify="right")
    table.add_column("share", justify="right")
    table.add_column("kernels", justify="right")
    table.add_column("top buckets")
    for name in ("prefill", "decode", "unknown"):
        entry = phase["phases"].get(name)
        if entry is None:
            continue
        tops = " ".join(
            f"{bucket['bucket']}={bucket['ms']:.1f}" for bucket in entry["buckets"][:3]
        )
        table.add_row(
            name, f"{entry['ms']:.1f}", f"{entry['share']:.1%}", f"{entry['kernels']:,}", tops
        )
    return table


def render_overhead(rows):
    table = Table(title="tracing overhead by arm")
    table.add_column("arm")
    table.add_column("runs", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("spread", justify="right")
    table.add_column("TPOT ms", justify="right")
    table.add_column("cost vs off", justify="right")
    for row in rows:
        table.add_row(
            row["arm"],
            str(row["runs"]),
            "-" if row["output_tokens_per_s"] is None else f"{row['output_tokens_per_s']:.1f}",
            "-" if row["spread"] is None else f"{row['spread']:.1f}",
            "-" if row["tpot_mean_ms"] is None else f"{row['tpot_mean_ms']:.2f}",
            "-" if row["throughput_cost"] is None else f"{row['throughput_cost']:+.1%}",
        )
    return table


def print_capture_analysis(record):
    """Print every view the capture actually supports, and say why one is missing."""
    if record["buckets"]:
        console.print(render_buckets(record))
        if record.get("gpu_active_share") is not None:
            console.print(f"[bold]GPU active:[/bold] {record['gpu_active_share']:.1%} of the window")
    for warning in record.get("warnings", []):
        console.print(f"[yellow]{warning}[/yellow]")

    layerwise = record.get("layerwise")
    if layerwise:
        console.print(
            f"[dim]resolved with a layer: {layerwise['n_resolved_with_layer']:,}"
            f"/{layerwise['n_kernels']:,}[/dim]"
        )
        console.print(render_layerwise_ops(layerwise))
        console.print(render_layers(layerwise))
    elif record.get("layerwise_skipped"):
        console.print("[dim]layer/op tables skipped (--no-layerwise)[/dim]")
    elif not record.get("has_trace", True):
        console.print("[dim]no trace in this directory — the untraced (`off`) arm[/dim]")
    else:
        console.print(
            "[dim]no NVTX ranges in this trace — capture with the `nvtx` arm for the "
            "layer/op breakdown[/dim]"
        )

    phase = record.get("phase")
    if phase and "phases" in phase:
        console.print(render_phase(phase))
        attribution = phase["attribution"]
        if attribution["direct_share"] is not None:
            console.print(
                f"[dim]phase named by the kernel itself: {attribution['direct_share']:.1%} "
                f"({attribution['n_anchors']:,} anchors); inferred from the nearest anchor, "
                f"median gap {attribution['gap_median_us']:.1f} us, "
                f"worst {attribution['gap_max_us'] / 1e3:.2f} ms[/dim]"
            )
    elif phase and phase.get("unavailable"):
        console.print(f"[yellow]prefill/decode split skipped: {phase['unavailable']}[/yellow]")
