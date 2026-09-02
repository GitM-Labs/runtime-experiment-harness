"""Tests for reading a capture directory back: buckets, layer/op, phase, arms.

Fixtures are written in the shape gitm actually writes -- trace.jsonl is JSONL with
a `_header` first line, kernels carry start_ns/end_ns and (only under --nvtx)
range_op/range_layer.
"""

import json

import pytest

from runtime_harness import analysis


def write_trace(path, kernels, header=True):
    lines = []
    if header:
        lines.append(json.dumps({"_header": {"workload_id": "vllm-serve", "duration_ns": 1_000_000}}))
    for kernel in kernels:
        lines.append(json.dumps({"kind": "kernel", "stream_id": 7, "device_id": 0, **kernel}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def nvtx_trace(path):
    """Two layers, three ops, plus one op that belongs to no layer."""
    return write_trace(
        path,
        [
            {"name": "fused_moe_kernel", "start_ns": 0, "end_ns": 300, "range_op": "moe_routed", "range_layer": 0},
            {"name": "nvjet_sm90", "start_ns": 300, "end_ns": 400, "range_op": "qkv_proj", "range_layer": 0},
            {"name": "fused_moe_kernel", "start_ns": 400, "end_ns": 600, "range_op": "moe_routed", "range_layer": 1},
            {"name": "layer_norm_fwd", "start_ns": 600, "end_ns": 650, "range_op": "input_layernorm", "range_layer": 1},
            # No layer: ops outside the decoder stack legitimately have none.
            {"name": "argmax", "start_ns": 650, "end_ns": 700, "range_op": "logits_processor", "range_layer": None},
        ],
    )


# --- layerwise --------------------------------------------------------------


def test_layerwise_splits_time_by_layer_and_op(tmp_path):
    summary = analysis.layerwise_summary(nvtx_trace(tmp_path / "trace.jsonl"))

    per_layer = {row["layer"]: row["ms"] for row in summary["per_layer"]}
    assert per_layer[0] == pytest.approx(400 / 1e6)
    assert per_layer[1] == pytest.approx(250 / 1e6)

    top_op = summary["per_op"][0]
    assert top_op["op"] == "moe_routed"
    assert top_op["kernels"] == 2
    assert top_op["layers"] == 2  # distinct layers the op appeared in


def test_an_op_with_no_layer_counts_in_the_op_table_only(tmp_path):
    """logits_processor and embed_tokens belong to no decoder layer. Dropping them
    would understate op time; putting them in the layer table would invent a layer."""
    summary = analysis.layerwise_summary(nvtx_trace(tmp_path / "trace.jsonl"))

    ops = {row["op"]: row for row in summary["per_op"]}
    assert ops["logits_processor"]["layers"] == 0
    assert all(row["layer"] is not None for row in summary["per_layer"])
    assert summary["n_resolved_with_layer"] == 4
    assert summary["n_kernels"] == 5


def test_a_trace_without_ranges_yields_no_layer_table(tmp_path):
    """Every arm except nvtx comes back with range_op null on every kernel. That is
    an absent measurement, not an empty one."""
    path = write_trace(
        tmp_path / "trace.jsonl",
        [{"name": "nvjet_sm90", "start_ns": 0, "end_ns": 100, "range_op": None, "range_layer": None}],
    )

    assert analysis.layerwise_summary(path) is None


def test_a_partial_trailing_line_does_not_lose_the_trace(tmp_path):
    """A process killed mid-write leaves half a record. Losing the last kernel beats
    failing the analysis of a multi-hour capture."""
    path = nvtx_trace(tmp_path / "trace.jsonl")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "kernel", "start_ns": 700, "end_n')

    assert analysis.layerwise_summary(path)["n_kernels"] == 5


def test_layerwise_of_a_missing_trace_is_none(tmp_path):
    assert analysis.layerwise_summary(tmp_path / "nope.jsonl") is None


# --- buckets ----------------------------------------------------------------


def test_bucket_rows_sort_by_time(tmp_path):
    rows = analysis.bucket_rows(
        {
            "buckets": [
                {"bucket": "gemm", "n_kernels": 10, "time_ns": 1_000, "share": 0.25},
                {"bucket": "moe", "n_kernels": 5, "time_ns": 3_000, "share": 0.75},
            ]
        }
    )

    assert [row["bucket"] for row in rows] == ["moe", "gemm"]
    assert rows[0]["time_s"] == pytest.approx(3e-6)


# --- serving summary --------------------------------------------------------


def test_throughput_is_read_from_the_server_block():
    """The client block is ServingSummary: latency percentiles and goodput, and no
    token counts at all. There is no client.output_tokens_per_s to read."""
    row = analysis.serving_row(
        {
            "tracing": "cupti",
            "server": {"output_tokens_per_s": 1234.5, "tpot_mean_s": 0.104},
            "client": {"n_requests": 32, "tpot_p50_s": 0.106},
        }
    )

    assert row["output_tokens_per_s"] == 1234.5
    assert row["tpot_mean_ms"] == pytest.approx(104.0)
    assert row["client_tpot_p50_ms"] == pytest.approx(106.0)


def test_a_summary_without_server_metrics_reports_none_not_zero():
    """A run whose /metrics scrape failed has an unknown throughput. Zero would drag
    an arm's median down and read as catastrophic overhead."""
    row = analysis.serving_row({"tracing": "off", "client": {"n_requests": 32}})

    assert row["output_tokens_per_s"] is None


# --- arms -------------------------------------------------------------------


def test_overhead_groups_on_the_reported_state_not_the_requested_one():
    """A directory named for the baseline that ran with the collector attached is
    the exact failure this table exists to expose."""
    rows = analysis.overhead_rows(
        [
            {"arm": "off", "serving": {"tracing": "cupti", "output_tokens_per_s": 900.0}},
            {"arm": "cupti", "serving": {"tracing": "cupti", "output_tokens_per_s": 910.0}},
        ]
    )

    assert [row["arm"] for row in rows] == ["cupti"]
    assert rows[0]["runs"] == 2


def test_overhead_prices_each_arm_against_the_untraced_baseline():
    rows = analysis.overhead_rows(
        [
            {"serving": {"tracing": "off", "output_tokens_per_s": 1000.0, "tpot_mean_ms": 100.0}},
            {"serving": {"tracing": "off", "output_tokens_per_s": 1000.0, "tpot_mean_ms": 100.0}},
            {"serving": {"tracing": "cupti", "output_tokens_per_s": 900.0, "tpot_mean_ms": 110.0}},
            {"serving": {"tracing": "cupti+nvtx", "output_tokens_per_s": 500.0, "tpot_mean_ms": 200.0}},
        ]
    )

    by_arm = {row["arm"]: row for row in rows}
    assert by_arm["off"]["throughput_cost"] == pytest.approx(0.0)
    assert by_arm["cupti"]["throughput_cost"] == pytest.approx(0.10)
    assert by_arm["cupti+nvtx"]["throughput_cost"] == pytest.approx(0.50)
    # off first, then increasing collection
    assert [row["arm"] for row in rows] == ["off", "cupti", "cupti+nvtx"]


def test_without_a_baseline_the_cost_column_is_blank_not_zero():
    """Two traced arms and no untraced one cannot price overhead at all."""
    rows = analysis.overhead_rows(
        [{"serving": {"tracing": "cupti", "output_tokens_per_s": 900.0}}]
    )

    assert rows[0]["throughput_cost"] is None


def test_runs_missing_throughput_are_counted_not_dropped():
    rows = analysis.overhead_rows(
        [
            {"serving": {"tracing": "cupti", "output_tokens_per_s": 900.0}},
            {"serving": {"tracing": "cupti", "output_tokens_per_s": None}},
        ]
    )

    assert rows[0]["runs"] == 2
    assert rows[0]["n_missing_throughput"] == 1
    assert rows[0]["output_tokens_per_s"] == 900.0


# --- discovery --------------------------------------------------------------


def test_the_untraced_arm_is_still_a_capture_directory(tmp_path):
    """--no-trace writes no trace.jsonl. Keying discovery on the trace would drop
    the one arm the comparison is measured against."""
    baseline = tmp_path / "ovh-off-1"
    baseline.mkdir()
    (baseline / "run_manifest.json").write_text("{}")

    assert analysis.is_capture_dir(baseline)


def test_a_parent_directory_expands_to_the_captures_inside_it(tmp_path):
    for name in ("run-a", "run-b"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "serving_summary.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("not a capture")

    found = analysis.discover_capture_dirs([tmp_path])

    assert [path.name for path in found] == ["run-a", "run-b"]


def test_overlapping_paths_are_not_analyzed_twice(tmp_path):
    directory = tmp_path / "run-a"
    directory.mkdir()
    (directory / "serving_summary.json").write_text("{}")

    assert len(analysis.discover_capture_dirs([tmp_path, directory])) == 1


# --- whole directory --------------------------------------------------------


def test_analyze_capture_reads_every_artifact(tmp_path):
    capture_dir = tmp_path / "run"
    capture_dir.mkdir()
    nvtx_trace(capture_dir / "trace.jsonl")
    (capture_dir / "kernel_breakdown.json").write_text(
        json.dumps(
            {
                "n_kernels": 5,
                "kernel_time_ns": 700,
                "gpu_active_share": 0.14,
                "buckets": [{"bucket": "moe", "n_kernels": 2, "time_ns": 500, "share": 0.71}],
                "warnings": ["GPU active only 14.0% of the window."],
            }
        )
    )
    (capture_dir / "serving_summary.json").write_text(
        json.dumps({"tracing": "cupti+nvtx", "server": {"output_tokens_per_s": 10.0}})
    )
    (capture_dir / "run_manifest.json").write_text(json.dumps({"served_model": "Qwen/X"}))

    record = analysis.analyze_capture(capture_dir, phase=False)

    assert record["served_model"] == "Qwen/X"
    assert record["gpu_active_share"] == 0.14
    assert record["serving"]["tracing"] == "cupti+nvtx"
    assert record["layerwise"]["n_resolved_with_layer"] == 4
    assert record["warnings"]


def test_an_untraced_directory_analyzes_without_a_trace(tmp_path):
    capture_dir = tmp_path / "off"
    capture_dir.mkdir()
    (capture_dir / "serving_summary.json").write_text(
        json.dumps({"tracing": "off", "server": {"output_tokens_per_s": 1200.0}})
    )

    record = analysis.analyze_capture(capture_dir)

    assert record["serving"]["output_tokens_per_s"] == 1200.0
    assert record["layerwise"] is None
    assert record["buckets"] == []


def test_phase_summary_says_why_it_is_missing(monkeypatch, tmp_path):
    """gitm owns the phase classifier. Without it importable the other two views
    still stand -- but the record must say the split was skipped, not that the
    capture had no phases."""
    path = nvtx_trace(tmp_path / "trace.jsonl")
    monkeypatch.setitem(__import__("sys").modules, "gitm.optimizer.deviation", None)

    result = analysis.phase_summary(path)

    assert "unavailable" in result


def test_the_three_reasons_for_no_layer_table_are_distinguishable(tmp_path, capsys):
    """"No trace at all", "a trace with no ranges" and "you asked me not to" read
    identically unless they are recorded separately -- and the fix for each is
    different."""
    untraced = tmp_path / "off"
    untraced.mkdir()
    (untraced / "run_manifest.json").write_text("{}")
    analysis.print_capture_analysis(analysis.analyze_capture(untraced))
    assert "untraced" in capsys.readouterr().out

    traced = tmp_path / "cupti"
    traced.mkdir()
    write_trace(
        traced / "trace.jsonl",
        [{"name": "nvjet", "start_ns": 0, "end_ns": 10, "range_op": None, "range_layer": None}],
    )
    analysis.print_capture_analysis(analysis.analyze_capture(traced, phase=False))
    assert "no NVTX ranges" in capsys.readouterr().out

    analysis.print_capture_analysis(
        analysis.analyze_capture(traced, layerwise=False, phase=False)
    )
    assert "--no-layerwise" in capsys.readouterr().out
