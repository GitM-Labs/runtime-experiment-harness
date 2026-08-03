"""Preflight checks for the GPU host: CUDA version and NVLink topology."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

console = Console()

# vLLM dropped support for CUDA older than this.
MIN_CUDA_VERSION = (13, 0)

# Model weights are large; keep them on the roomy workspace volume rather than
# the default ~/.cache/huggingface, which is often a small root disk.
DEFAULT_HF_HOME = Path("/workspace/hf_hub")
DEFAULT_EXPERIMENTS_DIR = Path("experiments")
DEFAULT_REPORTS_DIR = Path("/workspace/guidellm_reports")

# torch is listed alongside vllm so pip resolves them together -- vllm pins an
# exact torch build, and installing torch first would just get it replaced.
RUNTIME_PACKAGES = ["torch", "vllm", "guidellm", "huggingface-hub", "plotly"]


def format_version(version: tuple[int, int]) -> str:
    return ".".join(str(part) for part in version)


def run_command(command, capture_output=False, check=True, env=None):
    console.log(f"[blue]Running command:[/blue] {command}")
    process = subprocess.run(
        shlex.split(command),
        capture_output=capture_output,
        text=True,
        check=check,
        env=env,
    )
    return process.stdout if capture_output else None


def ensure_dependencies(packages=None):
    console.rule("Installing runtime dependencies")
    packages = list(packages) if packages is not None else RUNTIME_PACKAGES
    command = f"{sys.executable} -m pip install {' '.join(packages)}"
    run_command(command)
    console.log("[green]Dependencies installed successfully[/green]")


def prepare_workspace(hf_home=None, experiments_dir=None, reports_dir=None):
    """Create the HF cache, experiments, and report directories; point HF at the cache.

    The environment variables are exported into this process, so every vLLM and
    guidellm subprocess the harness launches inherits them. They do not leak back
    into the calling shell -- see the printed export line for that.
    """
    hf_home = Path(hf_home) if hf_home is not None else DEFAULT_HF_HOME
    experiments_dir = Path(experiments_dir) if experiments_dir is not None else DEFAULT_EXPERIMENTS_DIR
    reports_dir = Path(reports_dir) if reports_dir is not None else DEFAULT_REPORTS_DIR

    console.rule("Preparing workspace")

    flags = {"hf_home": "--hf-home", "experiments": "--experiments-dir", "reports": "--reports-dir"}
    for label, directory in (
        ("hf_home", hf_home),
        ("experiments", experiments_dir),
        ("reports", reports_dir),
    ):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Could not create {directory}: {exc}. "
                f"Pass {flags[label]} to use a writable location."
            ) from exc

    os.environ["HF_HOME"] = str(hf_home)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    console.print(f"[bold]HF_HOME:[/bold] {hf_home}")
    console.print(f"[bold]Experiments directory:[/bold] {experiments_dir.resolve()}")
    console.print(f"[bold]guidellm reports:[/bold] {reports_dir}")
    console.print(f"[dim]For your shell too:  export HF_HOME={hf_home}[/dim]")

    return hf_home, experiments_dir, reports_dir


def parse_cuda_version(smi_output: str):
    """Return the CUDA version reported by nvidia-smi as an (major, minor) tuple."""
    match = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", smi_output)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def ensure_cuda_supported():
    """Fail fast when the CUDA runtime is older than vLLM supports."""
    if shutil.which("nvidia-smi") is None:
        raise FileNotFoundError("nvidia-smi not found in PATH. NVIDIA drivers must be installed.")

    smi_output = run_command("nvidia-smi", capture_output=True) or ""
    version = parse_cuda_version(smi_output)
    minimum = format_version(MIN_CUDA_VERSION)

    if version is None:
        raise RuntimeError(
            f"Could not determine the CUDA version from nvidia-smi output. vLLM requires CUDA >= {minimum}."
        )

    detected = format_version(version)
    if version < MIN_CUDA_VERSION:
        raise RuntimeError(
            f"CUDA {detected} detected, but vLLM requires CUDA >= {minimum}. "
            "Upgrade the NVIDIA driver / CUDA toolkit before running experiments."
        )

    console.print(f"[bold]CUDA version:[/bold] {detected}")
    return version


def parse_gpu_topology():
    if shutil.which("nvidia-smi") is None:
        raise FileNotFoundError("nvidia-smi not found in PATH. NVIDIA drivers must be installed.")

    topo_raw = run_command("nvidia-smi topo --matrix", capture_output=True) or ""
    gpu_count = 0
    nvlink_matrix = []

    for line in topo_raw.splitlines():
        if line.startswith("GPU") and "CPU" not in line:
            parts = re.split(r"\s+", line.strip())
            if parts:
                gpu_count += 1
                nvlink_matrix.append(parts[1:gpu_count + 1])

    nvlink_available = any("NV" in cell.upper() for row in nvlink_matrix for cell in row)
    return gpu_count, nvlink_available, topo_raw


def run_preflight(install=True, hf_home=None, experiments_dir=None, reports_dir=None):
    """Verify the host and provision it for experiments.

    1. CUDA >= 13.0 (vLLM dropped everything older)
    2. GPU count and NVLink topology matrix
    3. torch / vllm / guidellm / huggingface-hub installed
    4. HF cache, experiments, and guidellm report directories created
    5. HF_HOME pointed at the cache

    Raises on the first unmet requirement.
    """
    ensure_cuda_supported()

    gpu_count, nvlink_available, topo_raw = parse_gpu_topology()
    console.print(f"[bold]Detected GPUs:[/bold] {gpu_count}")
    console.print(f"[bold]NVLink available:[/bold] {nvlink_available}")
    console.print("[bold]Topology matrix:[/bold]\n" + topo_raw)

    if install:
        ensure_dependencies()

    hf_home, experiments_dir, reports_dir = prepare_workspace(hf_home, experiments_dir, reports_dir)

    return gpu_count, nvlink_available, topo_raw, hf_home, experiments_dir, reports_dir
