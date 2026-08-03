import asyncio

import pytest

from runtime_harness import experiments
from runtime_harness.experiments import (
    Experiment,
    build_vllm_command,
    run_guidellm,
    server_is_ready,
    vllm_port,
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


def test_vllm_command_reads_manifest_parameters():
    exp = make_experiment(ep=2, vllm_args=["--gpu-memory-utilization", "0.95"])
    command = build_vllm_command(exp, 2)

    assert command[1:4] == ["-m", "vllm", "serve"]
    assert exp.model in command
    assert "--tensor-parallel-size" in command and "2" in command
    assert "--enable-expert-parallel" in command
    assert command[-2:] == ["--gpu-memory-utilization", "0.95"]


def test_manifest_tensor_parallel_size_is_not_duplicated():
    exp = make_experiment(vllm_args=["--tensor-parallel-size", "8"])
    assert build_vllm_command(exp, 8).count("--tensor-parallel-size") == 1


def test_expert_parallel_flag_is_not_duplicated():
    exp = make_experiment(ep=4, vllm_args=["--enable-expert-parallel"])
    assert build_vllm_command(exp, 4).count("--enable-expert-parallel") == 1


@pytest.mark.parametrize(
    "args,expected",
    [
        ([], 8000),
        (["--port", "8100"], 8100),
        (["--port=8200"], 8200),
        (["--gpu-memory-utilization", "0.9", "--port", "9001"], 9001),
    ],
)
def test_vllm_port_is_read_from_manifest(args, expected):
    assert vllm_port(make_experiment(vllm_args=args)) == expected


def test_server_is_ready_false_when_nothing_listening():
    # Port 1 is privileged and unbound in test environments.
    assert server_is_ready(1, timeout=0.2) is False


def test_start_vllm_server_reports_failure_when_process_dies(monkeypatch):
    class DeadProcess:
        stderr = type("S", (), {"read": staticmethod(lambda: "CUDA OOM")})()

        def poll(self):
            return 1

    monkeypatch.setattr(experiments.subprocess, "Popen", lambda *a, **kw: DeadProcess())
    result, process = asyncio.run(experiments.start_vllm_server(make_experiment(), 2))

    assert result["success"] is False
    assert "CUDA OOM" in result["stderr"]
    assert process is None


def test_start_vllm_server_returns_process_once_healthy(monkeypatch):
    class LiveProcess:
        stderr = None

        def poll(self):
            return None

    monkeypatch.setattr(experiments.subprocess, "Popen", lambda *a, **kw: LiveProcess())
    monkeypatch.setattr(experiments, "server_is_ready", lambda port, timeout=2.0: True)

    result, process = asyncio.run(experiments.start_vllm_server(make_experiment(vllm_args=["--port", "8123"]), 2))

    assert result["success"] is True
    assert result["port"] == 8123
    assert isinstance(process, LiveProcess)


def test_start_vllm_server_times_out_when_never_healthy(monkeypatch):
    terminated = []

    class HangingProcess:
        stderr = None

        def poll(self):
            return None

        def terminate(self):
            terminated.append(True)

    monkeypatch.setattr(experiments.subprocess, "Popen", lambda *a, **kw: HangingProcess())
    monkeypatch.setattr(experiments, "server_is_ready", lambda port, timeout=2.0: False)

    result, process = asyncio.run(experiments.start_vllm_server(make_experiment(), 2, startup_timeout=0))

    assert result["success"] is False
    assert "Timed out" in result["stderr"]
    assert process is None
    assert terminated == [True]


def test_run_guidellm_directs_report_into_reports_dir(tmp_path, monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(experiments.subprocess, "run", fake_run)
    result = run_guidellm(make_experiment(), tmp_path, "qwen-sxm-gpu2", "20260803T142305Z")

    expected = str(tmp_path / "20260803T142305Z_qwen-sxm-gpu2_guidellm.json")
    assert result["guidellm_output_path"] == expected
    assert "--output" in captured["command"]
    assert f"kind=json,path={expected}" in captured["command"]


def test_run_guidellm_respects_explicit_manifest_output(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(experiments.subprocess, "run", fake_run)
    exp = make_experiment(guidellm_command=["run", "--output", "kind=json,path=/tmp/mine.json"])
    result = run_guidellm(exp, tmp_path, "qwen-sxm-gpu2", "20260803T142305Z")

    assert result["guidellm_output_path"] == "/tmp/mine.json"
