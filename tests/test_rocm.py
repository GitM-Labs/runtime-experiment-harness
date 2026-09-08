"""ROCm backend: vendor detection, preflight package choice, rocprofv3 probing,
env passthrough. No GPU, no amd-smi — everything is patched at the seams."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime_harness import checks, rocm
from runtime_harness.experiments import expand_variants, load_experiments


# --------------------------------------------------------------------------- #
# vendor detection                                                            #
# --------------------------------------------------------------------------- #
def test_kfd_topology_wins_over_cli_presence(monkeypatch):
    """A box with kfd nodes is ROCm even if nvidia-smi is also on PATH
    (containers routinely carry a stub)."""
    monkeypatch.setattr(
        rocm.shutil, "which",
        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
    )
    monkeypatch.setattr(rocm, "Path", _FakePathFactory({"/sys/class/kfd/kfd/topology/nodes"}))
    assert rocm.is_rocm()


def test_is_rocm_false_on_plain_host(monkeypatch):
    monkeypatch.setattr(rocm.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(rocm, "Path", _FakePathFactory(exists_dirs=set()))
    assert not rocm.is_rocm()


def test_is_rocm_true_with_amd_smi_and_no_nvidia(monkeypatch):
    monkeypatch.setattr(rocm.shutil, "which", lambda name: "/usr/bin/amd-smi" if name == "amd-smi" else None)
    monkeypatch.setattr(rocm, "Path", _FakePathFactory(exists_dirs=set()))
    assert rocm.is_rocm()


class _FakePathFactory:
    """Minimal Path stand-in: is_dir() answers from a fixed set."""

    def __init__(self, exists_dirs):
        self.exists_dirs = exists_dirs

    def __call__(self, value):
        outer = self

        class _P:
            def is_dir(self_inner):
                return value in outer.exists_dirs

        return _P()


# --------------------------------------------------------------------------- #
# preflight package choice                                                    #
# --------------------------------------------------------------------------- #
def test_rocm_preflight_never_installs_torch_or_vllm(monkeypatch, tmp_path):
    """pip on a ROCm image would replace the shipped ROCm torch/vllm with CUDA
    wheels. The ROCm preflight must only install what the image lacks."""
    installed = []
    monkeypatch.setattr(checks, "ensure_dependencies", lambda pkgs=None: installed.append(pkgs))
    monkeypatch.setattr(rocm, "is_rocm", lambda: True)
    monkeypatch.setattr(rocm, "rocm_version", lambda: "7.0.0")
    monkeypatch.setattr(rocm, "parse_gpu_topology", lambda: (8, True, "XGMI matrix"))

    checks.run_preflight(
        install=True,
        hf_home=tmp_path / "hf",
        experiments_dir=tmp_path / "exp",
        reports_dir=tmp_path / "reports",
    )
    assert installed == [checks.ROCM_RUNTIME_PACKAGES]
    for forbidden in ("torch", "vllm"):
        assert forbidden not in checks.ROCM_RUNTIME_PACKAGES


def test_rocm_preflight_fails_without_visible_gpus(monkeypatch, tmp_path):
    monkeypatch.setattr(rocm, "is_rocm", lambda: True)
    monkeypatch.setattr(rocm, "rocm_version", lambda: "7.0.0")
    monkeypatch.setattr(rocm, "parse_gpu_topology", lambda: (0, False, ""))
    with pytest.raises(RuntimeError, match="no GPUs visible"):
        checks.run_preflight(install=False, hf_home=tmp_path, experiments_dir=tmp_path, reports_dir=tmp_path)


# --------------------------------------------------------------------------- #
# rocprofv3 wrapper                                                           #
# --------------------------------------------------------------------------- #
def test_rocprofv3_flags_are_probed_not_assumed(monkeypatch, tmp_path):
    """An older rocprofv3 without --rccl-trace must lose that plane, not kill
    the server launch with an unknown-flag error."""
    monkeypatch.setattr(rocm.shutil, "which", lambda name: "/usr/bin/rocprofv3")
    monkeypatch.setattr(
        rocm, "_rocprofv3_help",
        lambda: "--kernel-trace --hip-trace --output-format csv pftrace",
    )
    prefix = rocm.rocprofv3_prefix(tmp_path / "rocprof")
    assert "--kernel-trace" in prefix and "--hip-trace" in prefix
    assert "--rccl-trace" not in prefix
    assert "--pmc" not in prefix  # never on serving runs: counters serialize kernels
    assert prefix[-1] == "--"


def test_rocprofv3_disabled_by_escape_hatch(monkeypatch, tmp_path):
    monkeypatch.setenv(rocm.ENV_NO_ROCPROF, "1")
    monkeypatch.setattr(rocm.shutil, "which", lambda name: "/usr/bin/rocprofv3")
    assert rocm.rocprofv3_prefix(tmp_path) == []


# --------------------------------------------------------------------------- #
# manifest env passthrough                                                    #
# --------------------------------------------------------------------------- #
def test_env_survives_loading_and_variant_merge(tmp_path):
    config = tmp_path / "experiments.yaml"
    config.write_text(
        json.dumps({
            "gpu_counts": [8],
            "experiments": [{
                "name": "kimi",
                "model": "moonshotai/Kimi-K2.5",
                "prompt": "hi",
                "env": {"VLLM_ROCM_USE_AITER": "1"},
                "variants": [
                    {"name": "base"},
                    {"name": "rccl-log", "env": {"RCCL_DEBUG": "INFO"}},
                ],
            }],
        }),
        encoding="utf-8",
    )
    _counts, experiments = load_experiments(config)
    by_name = {exp.name: exp for exp in experiments}
    assert by_name["kimi-base"].env == {"VLLM_ROCM_USE_AITER": "1"}
    assert by_name["kimi-rccl-log"].env == {
        "VLLM_ROCM_USE_AITER": "1",
        "RCCL_DEBUG": "INFO",
    }


def test_variant_env_overrides_base(tmp_path):
    entry = {
        "name": "power",
        "model": "m",
        "prompt": "p",
        "env": {"A": "1"},
        "variants": [{"name": "capped", "env": {"A": "2"}}],
    }
    (expanded,) = expand_variants(entry)
    assert expanded["env"] == {"A": "2"}
