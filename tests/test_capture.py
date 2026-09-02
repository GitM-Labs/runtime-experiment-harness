"""Tests for the `gitm capture serve` path: arms, command building, and guards.

No capture is run. What is checked is the command the harness builds and the
conditions it refuses to start under -- the failures here cost an hour of GPU time
each to discover any other way.
"""

from pathlib import Path

import pytest

from runtime_harness import capture
from runtime_harness.capture import CaptureSpec, build_capture_command, capture_spec_from_raw


def command_for(arm, **spec_kwargs):
    spec = CaptureSpec(**spec_kwargs)
    return build_capture_command(spec, arm, Path("/w/out"), "Qwen/Qwen3.6-35B-A3B", ["--enforce-eager"])


# --- arms -------------------------------------------------------------------


def test_each_arm_maps_to_its_own_collector_flags():
    assert "--no-trace" in command_for("off")
    assert "--nvtx" in command_for("nvtx")

    plain = command_for("cupti")
    assert "--no-trace" not in plain and "--nvtx" not in plain


def test_the_manifest_never_sets_the_layerwise_flag_itself():
    """--enable-layerwise-nvtx-tracing is added by gitm's own --nvtx. Setting it in
    vllm_args would push ranges in EVERY arm, including the ones whose throughput is
    supposed to be range-free -- the cost lands in the comparison while appearing
    nowhere in the configuration."""
    for arm in ("off", "cupti", "nvtx"):
        assert "--enable-layerwise-nvtx-tracing" not in command_for(arm)


def test_an_unknown_arm_is_rejected_at_load_not_at_launch():
    with pytest.raises(ValueError, match="Unknown capture arm"):
        capture_spec_from_raw({"arms": ["cudagraphs"]})


def test_arms_are_interleaved_across_repetitions():
    """Three offs then three cuptis donates any session-long drift -- thermals, a
    co-tenant -- entirely to whichever arm ran during it."""
    spec = CaptureSpec(arms=["off", "cupti", "nvtx"], repeat=2)

    assert spec.runs() == [
        ("off", 1), ("cupti", 1), ("nvtx", 1),
        ("off", 2), ("cupti", 2), ("nvtx", 2),
    ]


# --- command construction ---------------------------------------------------


def test_the_serve_command_is_passed_through_after_the_separator():
    command = command_for("cupti")
    separator = command.index("--")

    assert command[separator + 1:] == ["vllm", "serve", "Qwen/Qwen3.6-35B-A3B", "--enforce-eager"]


def test_load_shape_reaches_gitm():
    command = command_for(
        "cupti", requests=64, concurrency=8, input_tokens=512, output_tokens=512, warmup=8
    )

    for flag, value in (
        ("--requests", "64"),
        ("--concurrency", "8"),
        ("--input-tokens", "512"),
        ("--output-tokens", "512"),
        ("--warmup", "8"),
    ):
        assert command[command.index(flag) + 1] == value


def test_optional_timeouts_are_omitted_rather_than_defaulted():
    """gitm's own defaults (1800s health, 600s request) are tuned for a cold weight
    download; restating them here would silently pin them to whatever this file
    happened to say."""
    command = command_for("cupti")
    assert "--health-timeout" not in command

    assert "--health-timeout" in command_for("cupti", health_timeout=900.0)


def test_extra_args_land_before_the_separator():
    """After `--` they would be read as vLLM flags and vLLM would reject them."""
    command = command_for("cupti", extra_args=["--metrics-interval", "0.5"])

    assert command.index("--metrics-interval") < command.index("--")


# --- guards -----------------------------------------------------------------


def test_keep_server_is_refused_when_another_run_follows():
    """A kept server holds the port and ~133 GiB of KV cache at 0.95 utilization.
    The next arm then fails inside engine startup with a message naming neither
    memory nor the previous run, and every arm after the first is recorded as a
    failure while the results file still looks complete."""
    spec = CaptureSpec(keep_server=True, arms=["cupti", "nvtx"])

    with pytest.raises(ValueError, match="keep_server"):
        capture.check_keep_server(spec, total_runs=2)

    capture.check_keep_server(spec, total_runs=1)  # the last run may keep its server


def test_unknown_capture_settings_are_named():
    """A typo'd key would otherwise be dropped and the run would proceed with
    gitm's default in its place."""
    with pytest.raises(ValueError, match="Unknown capture settings: conccurency"):
        capture_spec_from_raw({"conccurency": 8})


def test_capture_true_means_all_defaults():
    assert capture_spec_from_raw(True).arms == ["cupti"]
    assert capture_spec_from_raw(None) is None


def test_repeat_must_be_positive():
    with pytest.raises(ValueError, match="repeat"):
        capture_spec_from_raw({"repeat": 0})


def test_missing_hf_token_is_warned_about(monkeypatch, capsys):
    """A gated checkpoint fails during weight download -- minutes in, after
    preflight has already passed."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    capture.warn_if_no_hf_token()
    assert "HF_TOKEN" in capsys.readouterr().out

    monkeypatch.setenv("HF_TOKEN", "hf_x")
    capture.warn_if_no_hf_token()
    assert "HF_TOKEN" not in capsys.readouterr().out


# --- running ----------------------------------------------------------------


def test_gitm_exit_codes_are_preserved_as_status(monkeypatch, tmp_path):
    """0 captured / 1 failed / 2 skipped-by-preflight are three different outcomes.
    Flattened to a boolean, a sweep that captured nothing on an unsupported box
    reads the same as one that crashed."""
    class FakeProcess:
        def __init__(self, code):
            self.stdout = iter(["==> out dir: x\n"])
            self._code = code

        def wait(self):
            return self._code

    for code, status in ((0, "captured"), (1, "failed"), (2, "skipped")):
        monkeypatch.setattr(
            capture.subprocess, "Popen", lambda *a, code=code, **kw: FakeProcess(code)
        )
        record = capture.run_capture(
            CaptureSpec(), "cupti", tmp_path / f"out{code}", "m", ["--x"]
        )
        assert record["status"] == status
        assert record["success"] is (code == 0)
        assert record["returncode"] == code


def test_a_missing_gitm_says_how_to_install_it(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise FileNotFoundError("gitm")

    monkeypatch.setattr(capture.subprocess, "Popen", boom)

    with pytest.raises(RuntimeError, match="rex install"):
        capture.run_capture(CaptureSpec(), "cupti", tmp_path / "out", "m")


def test_gpu_count_masks_the_visible_devices(monkeypatch, tmp_path):
    """gitm sizes tensor parallelism off CUDA_VISIBLE_DEVICES, not off nvidia-smi:
    asking for 8 ranks on a box where CUDA exposes 2 fails deep inside distributed
    init rather than at the flag."""
    seen = {}

    class FakeProcess:
        stdout = iter([])

        def wait(self):
            return 0

    def fake_popen(command, **kwargs):
        seen.update(kwargs.get("env") or {})
        return FakeProcess()

    monkeypatch.setattr(capture.subprocess, "Popen", fake_popen)
    capture.run_capture(CaptureSpec(), "cupti", tmp_path / "out", "m", gpu_count=4)

    assert seen["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
