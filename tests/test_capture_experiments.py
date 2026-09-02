"""Manifest-level wiring for capture experiments, and the `analyze`/`compare` CLI."""

import json
from pathlib import Path

import pytest

from runtime_harness import cli, experiments
from runtime_harness.capture import CaptureSpec
from runtime_harness.experiments import Experiment, load_experiments


CAPTURE_MANIFEST = """
gpu_counts: [1]
experiments:
  - name: qwen-eager
    model: Qwen/Qwen3.6-35B-A3B
    vllm_args: ["--enforce-eager"]
    capture:
      requests: 64
      concurrency: 8
      arms: ["cupti", "nvtx"]
"""


def write(tmp_path, text, name="e.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- manifest ---------------------------------------------------------------


def test_a_capture_entry_needs_no_prompt(tmp_path):
    """guidellm needs a prompt; gitm builds its own from --input-tokens. Requiring
    one here would be a field nobody reads."""
    _, loaded = load_experiments(write(tmp_path, CAPTURE_MANIFEST))

    assert loaded[0].capture.arms == ["cupti", "nvtx"]
    assert loaded[0].prompt == ""


def test_a_guidellm_entry_still_requires_its_prompt(tmp_path):
    manifest = "gpu_counts: [1]\nexperiments:\n  - name: n\n    model: m\n"

    with pytest.raises(ValueError, match="needs a prompt"):
        load_experiments(write(tmp_path, manifest))


def test_a_variant_overrides_single_capture_settings(tmp_path):
    """The slow arm may want its own window without restating the block -- and
    without the load shape silently differing between arms."""
    manifest = """
gpu_counts: [1]
experiments:
  - name: sweep
    model: m
    capture: {requests: 128, concurrency: 32, arms: ["cupti"]}
    variants:
      - name: graphs
      - name: eager
        vllm_args: ["--enforce-eager"]
        capture: {arms: ["cupti", "nvtx"]}
"""
    _, loaded = load_experiments(write(tmp_path, manifest))

    graphs, eager = loaded
    assert graphs.capture.arms == ["cupti"]
    assert eager.capture.arms == ["cupti", "nvtx"]
    # the shared load shape survives the override
    assert eager.capture.requests == 128 and eager.capture.concurrency == 32


# --- run loop ---------------------------------------------------------------


def capture_experiment(**capture_kwargs):
    return Experiment(
        name="qwen",
        model="Qwen/X",
        prompt="",
        batch_size=1, sequence_length=1, num_steps=1, ep=1, dp=1, tp=1, pp=1,
        vllm_args=["--enforce-eager"],
        capture=CaptureSpec(**capture_kwargs),
    )


@pytest.fixture
def no_gpu_waits(monkeypatch):
    monkeypatch.setattr(experiments, "wait_for_idle_vram", lambda **kw: True)


def test_every_arm_and_repetition_gets_its_own_directory(no_gpu_waits, monkeypatch, tmp_path):
    """Arms sharing an --out would each overwrite the last, and the run directory
    would still look complete while holding one arm's trace under every arm's name."""
    seen = []

    def fake_run_capture(spec, arm, out_dir, model, vllm_args, gpu_count=None, env=None):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        seen.append(Path(out_dir).name)
        return {"name": Path(out_dir).name, "engine": "gitm-capture", "arm": arm,
                "success": True, "status": "captured", "capture_dir": str(out_dir)}

    monkeypatch.setattr(experiments, "run_capture", fake_run_capture)
    monkeypatch.setattr(experiments, "analyze_capture", lambda *a, **kw: {"serving": None})
    monkeypatch.setattr(experiments, "print_capture_analysis", lambda record: None)

    records = experiments.run_capture_experiment(
        capture_experiment(arms=["off", "cupti"], repeat=2), 1, tmp_path, timestamp="T"
    )

    assert len(records) == 4
    assert len(set(seen)) == 4
    assert seen[0] == "T_qwen-gpu1_off-r1"


def test_a_single_arm_run_is_not_labelled_r1(no_gpu_waits, monkeypatch, tmp_path):
    monkeypatch.setattr(
        experiments, "run_capture",
        lambda spec, arm, out_dir, *a, **kw: (
            Path(out_dir).mkdir(parents=True, exist_ok=True),
            {"name": "x", "engine": "gitm-capture", "arm": arm, "success": False,
             "status": "failed", "capture_dir": str(out_dir)},
        )[1],
    )

    records = experiments.run_capture_experiment(
        capture_experiment(arms=["cupti"]), 1, tmp_path, timestamp="T"
    )

    assert Path(records[0]["capture_dir"]).name == "T_qwen-gpu1_cupti"


def test_an_arm_that_ran_in_the_wrong_tracing_state_is_flagged(no_gpu_waits, monkeypatch, tmp_path):
    """An injection variable inherited from the shell attaches the collector to the
    baseline arm. That arm still reports a throughput number, and the number reads
    as "collection is free" rather than as a broken baseline."""
    monkeypatch.setattr(
        experiments, "run_capture",
        lambda spec, arm, out_dir, *a, **kw: (
            Path(out_dir).mkdir(parents=True, exist_ok=True),
            {"name": "x", "engine": "gitm-capture", "arm": arm, "success": True,
             "status": "captured", "capture_dir": str(out_dir)},
        )[1],
    )
    monkeypatch.setattr(
        experiments, "analyze_capture",
        lambda *a, **kw: {"serving": {"tracing": "cupti"}},
    )
    monkeypatch.setattr(experiments, "print_capture_analysis", lambda record: None)

    records = experiments.run_capture_experiment(
        capture_experiment(arms=["off"]), 1, tmp_path, timestamp="T"
    )

    assert records[0]["arm_mismatch"] == {"expected": "off", "reported": "cupti"}


def test_a_skipped_capture_is_not_analyzed(no_gpu_waits, monkeypatch, tmp_path):
    """Exit 2 means preflight refused to start anything. There is no directory to
    read, and analyzing it would report an empty capture as a real one."""
    monkeypatch.setattr(
        experiments, "run_capture",
        lambda spec, arm, out_dir, *a, **kw: (
            Path(out_dir).mkdir(parents=True, exist_ok=True),
            {"name": "x", "engine": "gitm-capture", "arm": arm, "success": False,
             "status": "skipped", "capture_dir": str(out_dir)},
        )[1],
    )
    monkeypatch.setattr(
        experiments, "analyze_capture",
        lambda *a, **kw: pytest.fail("a skipped capture must not be analyzed"),
    )

    records = experiments.run_capture_experiment(
        capture_experiment(arms=["cupti"]), 1, tmp_path, timestamp="T"
    )

    assert "analysis" not in records[0]


def test_the_harness_record_lands_beside_the_capture(no_gpu_waits, monkeypatch, tmp_path):
    """So a capture directory carries which arm and which sweep it belongs to --
    gitm's own artifacts know nothing about the manifest."""
    monkeypatch.setattr(
        experiments, "run_capture",
        lambda spec, arm, out_dir, *a, **kw: (
            Path(out_dir).mkdir(parents=True, exist_ok=True),
            {"name": "x", "engine": "gitm-capture", "arm": arm, "success": False,
             "status": "failed", "capture_dir": str(out_dir)},
        )[1],
    )

    records = experiments.run_capture_experiment(
        capture_experiment(arms=["nvtx"]), 2, tmp_path, timestamp="T"
    )

    saved = json.loads(
        (Path(records[0]["capture_dir"]) / "harness_record.json").read_text()
    )
    assert saved["arm"] == "nvtx"
    assert saved["gpu_count"] == 2


# --- CLI --------------------------------------------------------------------


def make_capture_dir(path, tracing, tokens_per_s):
    path.mkdir(parents=True, exist_ok=True)
    (path / "run_manifest.json").write_text(json.dumps({"served_model": "Qwen/X"}))
    (path / "serving_summary.json").write_text(
        json.dumps({"tracing": tracing, "server": {"output_tokens_per_s": tokens_per_s}})
    )
    return path


def test_compare_prints_a_row_per_arm(tmp_path, capsys):
    make_capture_dir(tmp_path / "ovh-off-1", "off", 1000.0)
    make_capture_dir(tmp_path / "ovh-cupti-1", "cupti", 900.0)

    assert cli.main(["compare", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "off" in out and "cupti" in out


def test_compare_writes_json_when_asked(tmp_path):
    make_capture_dir(tmp_path / "run", "cupti", 900.0)
    destination = tmp_path / "out" / "overhead.json"

    assert cli.main(["compare", str(tmp_path), "--json", str(destination)]) == 0
    assert json.loads(destination.read_text())[0]["arm"] == "cupti"


def test_analyze_reports_nothing_found_rather_than_crashing(tmp_path, capsys):
    assert cli.main(["analyze", str(tmp_path)]) == 1
    assert "No capture directories" in capsys.readouterr().out


def test_analyze_writes_its_records(tmp_path):
    make_capture_dir(tmp_path / "run", "cupti", 900.0)
    destination = tmp_path / "analysis.json"

    assert cli.main(["analyze", str(tmp_path), "--json", str(destination)]) == 0
    records = json.loads(destination.read_text())
    assert records[0]["served_model"] == "Qwen/X"


# --- templates --------------------------------------------------------------


def test_every_bundled_template_ships_and_parses(tmp_path):
    """A template that does not load is worse than no template: it is discovered
    after `rex init` has already written it into someone's workspace."""
    from runtime_harness.experiments import TEMPLATES

    for name in TEMPLATES:
        destination = tmp_path / f"{name}.yaml"
        assert cli.main(["init", str(destination), "--template", name]) == 0
        gpu_counts, loaded = load_experiments(destination)
        assert gpu_counts and loaded


def test_the_capture_templates_are_capture_driven(tmp_path):
    for name, arms in (
        ("capture", {"cupti", "nvtx"}),
        ("overhead", {"off", "cupti", "nvtx"}),
        ("cudagraph-spec-capture", {"cupti", "nvtx"}),
    ):
        destination = tmp_path / f"{name}.yaml"
        cli.main(["init", str(destination), "--template", name])
        _, loaded = load_experiments(destination)

        assert all(entry.capture is not None for entry in loaded)
        assert {arm for entry in loaded for arm in entry.capture.arms} <= arms


def test_no_capture_template_pushes_nvtx_ranges_by_hand(tmp_path):
    """--enable-layerwise-nvtx-tracing in vllm_args would push ranges in every arm,
    including the ones whose throughput must stay range-free. gitm's --nvtx adds it
    to exactly the arm that collects them."""
    from runtime_harness.experiments import TEMPLATES

    for name in TEMPLATES:
        destination = tmp_path / f"{name}.yaml"
        cli.main(["init", str(destination), "--template", name])
        _, loaded = load_experiments(destination)

        for entry in loaded:
            assert "--enable-layerwise-nvtx-tracing" not in entry.vllm_args


def test_an_unknown_template_names_the_available_ones(tmp_path, capsys):
    """argparse rejects it at the flag; template_path is the guard for callers that
    reach past the CLI."""
    from runtime_harness.experiments import template_path

    with pytest.raises(SystemExit):
        cli.main(["init", str(tmp_path / "e.yaml"), "--template", "nope"])
    assert "invalid choice" in capsys.readouterr().err

    with pytest.raises(ValueError, match="Available: capture"):
        template_path("nope")


# --- the whole sweep --------------------------------------------------------


def run_manifest(manifest, tmp_path, monkeypatch, capture_status="captured"):
    """Drive run_all_experiments with the GPU probe and gitm both stubbed out."""
    import asyncio

    monkeypatch.setattr(
        experiments,
        "run_preflight",
        lambda **kw: (1, True, "topo", kw["hf_home"], Path(kw["experiments_dir"]), kw["reports_dir"]),
    )
    monkeypatch.setattr(experiments, "wait_for_idle_vram", lambda **kw: True)
    monkeypatch.setattr(experiments, "print_capture_analysis", lambda record: None)
    calls = []

    def fake_run_capture(spec, arm, out_dir, model, vllm_args, gpu_count=None, env=None):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        calls.append({"arm": arm, "model": model, "vllm_args": vllm_args, "out_dir": Path(out_dir)})
        return {"name": Path(out_dir).name, "engine": "gitm-capture", "arm": arm,
                "success": capture_status == "captured", "status": capture_status,
                "capture_dir": str(out_dir)}

    monkeypatch.setattr(experiments, "run_capture", fake_run_capture)
    monkeypatch.setattr(
        experiments, "analyze_capture",
        lambda out_dir, **kw: {"serving": {"tracing": "cupti", "output_tokens_per_s": 900.0}},
    )

    config = write(tmp_path, manifest)
    results = asyncio.run(
        experiments.run_all_experiments(
            config, install=False, output_dir=tmp_path, experiments_dir=tmp_path / "experiments"
        )
    )
    return results, calls


def test_a_capture_sweep_never_starts_a_server_itself(tmp_path, monkeypatch):
    """The harness starting vLLM would produce an untraceable server: the driver
    reads CUDA_INJECTION64_PATH once, at CUDA init."""
    def fail(*args, **kwargs):
        pytest.fail("a capture experiment must not go through start_vllm_server")

    monkeypatch.setattr(experiments, "start_vllm_server", fail)

    results, calls = run_manifest(CAPTURE_MANIFEST, tmp_path, monkeypatch)

    assert [call["arm"] for call in calls] == ["cupti", "nvtx"]
    assert calls[0]["vllm_args"] == ["--enforce-eager"]
    assert all(record["engine"] == "gitm-capture" for record in results)


def test_the_sweep_writes_its_results_file(tmp_path, monkeypatch):
    run_manifest(CAPTURE_MANIFEST, tmp_path, monkeypatch)

    saved = json.loads((tmp_path / "experiment_results.json").read_text())
    assert {record["arm"] for record in saved} == {"cupti", "nvtx"}


def test_keep_server_is_refused_before_anything_launches(tmp_path, monkeypatch):
    """An hour into a sweep is the wrong time to discover that arm two could never
    have started."""
    manifest = CAPTURE_MANIFEST.replace(
        'arms: ["cupti", "nvtx"]', 'arms: ["cupti", "nvtx"]\n      keep_server: true'
    )

    with pytest.raises(ValueError, match="keep_server"):
        run_manifest(manifest, tmp_path, monkeypatch)

    assert not (tmp_path / "experiments").exists()
