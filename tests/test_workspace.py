import json
import re

import pytest

from runtime_harness import checks
from runtime_harness.experiments import (
    Experiment,
    experiment_id,
    guidellm_output_path,
    save_experiment_result,
    utc_timestamp,
)


def make_experiment(**overrides):
    defaults = dict(
        name="qwen-sxm",
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        prompt="hello",
        batch_size=1,
        sequence_length=512,
        num_steps=10,
        ep=1,
        dp=1,
        tp=2,
        pp=1,
    )
    defaults.update(overrides)
    return Experiment(**defaults)


def test_prepare_workspace_creates_dirs_and_sets_hf_home(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    hf_home = tmp_path / "hf_hub"
    experiments = tmp_path / "experiments"
    reports = tmp_path / "guidellm_reports"

    returned = checks.prepare_workspace(hf_home, experiments, reports)

    assert hf_home.is_dir()
    assert experiments.is_dir()
    assert reports.is_dir()
    assert checks.os.environ["HF_HOME"] == str(hf_home)
    assert returned == (hf_home, experiments, reports)


def test_prepare_workspace_is_idempotent(tmp_path):
    args = (tmp_path / "hf_hub", tmp_path / "experiments", tmp_path / "reports")
    (tmp_path / "hf_hub").mkdir()
    checks.prepare_workspace(*args)
    checks.prepare_workspace(*args)


def test_prepare_workspace_reports_unwritable_location(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file")
    with pytest.raises(RuntimeError, match=r"Could not create.*--hf-home"):
        checks.prepare_workspace(blocker / "hf_hub", tmp_path / "experiments", tmp_path / "reports")


def test_prepare_workspace_names_the_failing_directory(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file")
    with pytest.raises(RuntimeError, match=r"--reports-dir"):
        checks.prepare_workspace(tmp_path / "hf", tmp_path / "exp", blocker / "reports")


def test_default_hf_home_is_workspace_path():
    assert str(checks.DEFAULT_HF_HOME) == "/workspace/hf_hub"


def test_default_reports_dir_is_workspace_path():
    assert str(checks.DEFAULT_REPORTS_DIR) == "/workspace/guidellm_reports"


def test_runtime_packages_cover_requested_stack():
    assert {"torch", "vllm", "guidellm", "huggingface-hub"} <= set(checks.RUNTIME_PACKAGES)


def test_timestamp_is_filename_safe():
    stamp = utc_timestamp()
    assert re.fullmatch(r"\d{8}T\d{6}Z", stamp)
    assert ":" not in stamp


def test_experiment_id_includes_gpu_count():
    assert experiment_id(make_experiment(), 4) == "qwen-sxm-gpu4"


def test_save_experiment_result_uses_requested_layout(tmp_path):
    result = {"name": "qwen-sxm", "engine": "guidellm", "success": True, "stdout": "latency: 12.5"}
    path = save_experiment_result(result, tmp_path, "qwen-sxm-gpu2", timestamp="20260803T142305Z")

    assert path == tmp_path / "qwen-sxm-gpu2" / "20260803T142305Z_qwen-sxm-gpu2.json"
    payload = json.loads(path.read_text())
    assert payload["experiment_id"] == "qwen-sxm-gpu2"
    assert payload["timestamp"] == "20260803T142305Z"
    assert payload["latency"] == 12.5


def test_save_experiment_result_copies_guidellm_report(tmp_path):
    report = tmp_path / "run.json"
    report.write_text('{"benchmarks": []}')
    result = {
        "name": "qwen-sxm",
        "engine": "guidellm",
        "success": True,
        "stdout": "",
        "guidellm_output_path": str(report),
    }

    save_experiment_result(result, tmp_path / "experiments", "qwen-sxm-gpu2", timestamp="20260803T142305Z")

    copied = tmp_path / "experiments" / "qwen-sxm-gpu2" / "20260803T142305Z_qwen-sxm-gpu2_guidellm.json"
    assert json.loads(copied.read_text()) == {"benchmarks": []}


def test_save_experiment_result_tolerates_missing_report(tmp_path):
    result = {"name": "x", "engine": "guidellm", "success": True, "stdout": "", "guidellm_output_path": "/nope.json"}
    path = save_experiment_result(result, tmp_path, "x-gpu1")
    assert path.exists()


def test_guidellm_output_path_extracted_from_command():
    exp = make_experiment(
        guidellm_command=["run", "--output", "kind=json,path=/workspace/tp2-run.json"]
    )
    assert guidellm_output_path(exp) == "/workspace/tp2-run.json"


def test_guidellm_output_path_absent_returns_none():
    assert guidellm_output_path(make_experiment()) is None
