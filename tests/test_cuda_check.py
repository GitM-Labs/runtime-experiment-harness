import pytest

from runtime_harness import checks


SMI_TEMPLATE = """\
Mon Aug  3 10:00:00 2026
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 580.65.06    Driver Version: 580.65.06    CUDA Version: {version}  |
|-------------------------------+----------------------+----------------------+
"""


def test_parse_cuda_version_reads_smi_header():
    assert checks.parse_cuda_version(SMI_TEMPLATE.format(version="13.0")) == (13, 0)


def test_parse_cuda_version_returns_none_when_absent():
    assert checks.parse_cuda_version("no gpu here") is None


def _patch_smi(monkeypatch, output):
    monkeypatch.setattr(checks.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(checks, "run_command", lambda *a, **kw: output)


def test_ensure_cuda_supported_accepts_minimum(monkeypatch):
    _patch_smi(monkeypatch, SMI_TEMPLATE.format(version="13.0"))
    assert checks.ensure_cuda_supported() == (13, 0)


def test_ensure_cuda_supported_accepts_newer(monkeypatch):
    _patch_smi(monkeypatch, SMI_TEMPLATE.format(version="13.2"))
    assert checks.ensure_cuda_supported() == (13, 2)


def test_ensure_cuda_supported_rejects_older(monkeypatch):
    _patch_smi(monkeypatch, SMI_TEMPLATE.format(version="12.8"))
    with pytest.raises(RuntimeError, match="requires CUDA >= 13.0"):
        checks.ensure_cuda_supported()


def test_ensure_cuda_supported_rejects_unparseable(monkeypatch):
    _patch_smi(monkeypatch, "garbage output")
    with pytest.raises(RuntimeError, match="Could not determine the CUDA version"):
        checks.ensure_cuda_supported()


def test_ensure_cuda_supported_requires_nvidia_smi(monkeypatch):
    monkeypatch.setattr(checks.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError):
        checks.ensure_cuda_supported()
