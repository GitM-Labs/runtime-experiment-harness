"""Experiment orchestration: launch vLLM, drive guidellm, collect results."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:  # pragma: no cover - handled when optional plotting dependency is absent.
    go = None

import yaml
from rich.console import Console

from . import __version__
from .banner import harness_banner
from .checks import DEFAULT_EXPERIMENTS_DIR, run_preflight

console = Console()

# Config and outputs resolve against the invocation directory, never the install
# location -- site-packages is often read-only and is never the user's workspace.
DEFAULT_CONFIG_NAME = "experiments.yaml"
BUNDLED_CONFIG = Path(__file__).parent / "data" / DEFAULT_CONFIG_NAME


def utc_timestamp() -> str:
    """Filename-safe UTC stamp: 20260803T142305Z.

    Deliberately not literal `date:time` -- colons are illegal in filenames on
    macOS and Windows and need quoting in every shell command that touches them.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def experiment_id(experiment: "Experiment", gpu_count: int) -> str:
    """Stable identifier for one (experiment, GPU count) pair in the sweep."""
    return f"{experiment.name}-gpu{gpu_count}"


def save_experiment_result(result: dict, experiments_dir: Path, exp_id: str, timestamp=None) -> Path:
    """Write one guidellm result to experiments/<id>/<timestamp>_<id>.json.

    If guidellm was configured to emit its own JSON report (via
    `--output kind=json,path=...`), that artifact is copied in alongside so the
    run directory is self-contained.
    """
    timestamp = timestamp or utc_timestamp()
    run_dir = Path(experiments_dir) / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    output_file = run_dir / f"{timestamp}_{exp_id}.json"
    payload = dict(result)
    payload["experiment_id"] = exp_id
    payload["timestamp"] = timestamp
    payload["latency"] = parse_latency_from_output(result.get("stdout", ""))

    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    guidellm_report = result.get("guidellm_output_path")
    if guidellm_report and Path(guidellm_report).is_file():
        shutil.copyfile(guidellm_report, run_dir / f"{timestamp}_{exp_id}_guidellm.json")

    console.log(f"[green]Saved {exp_id} result to {output_file}[/green]")
    return output_file


def guidellm_output_path(experiment: "Experiment"):
    """Extract the report path guidellm was told to write, if any."""
    for arg in list(experiment.guidellm_command) + list(experiment.guidellm_args):
        match = re.search(r"kind=json,\s*path=([^,\s]+)", arg)
        if match:
            return match.group(1)
    return None


@dataclasses.dataclass
class Experiment:
    name: str
    model: str
    prompt: str
    batch_size: int
    sequence_length: int
    num_steps: int
    ep: int
    dp: int
    tp: int
    pp: int
    vllm_args: list[str] = dataclasses.field(default_factory=list)
    guidellm_args: list[str] = dataclasses.field(default_factory=list)
    guidellm_command: list[str] = dataclasses.field(default_factory=list)
    metadata: dict = dataclasses.field(default_factory=dict)


def normalize_arg_list(raw):
    if raw is None:
        return []
    if isinstance(raw, str):
        return shlex.split(raw)
    if isinstance(raw, list):
        return [str(item) for item in raw]
    raise ValueError("Argument lists must be strings or lists.")


def load_experiments(config_path: Path):
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    gpu_counts = config.get("gpu_counts", [2, 4, 8])
    experiments = []
    for raw in config.get("experiments", []):
        experiment = Experiment(
            name=raw["name"],
            model=raw["model"],
            prompt=raw["prompt"],
            batch_size=int(raw.get("batch_size", 1)),
            sequence_length=int(raw.get("sequence_length", 512)),
            num_steps=int(raw.get("num_steps", 10)),
            ep=int(raw.get("ep", 1)),
            dp=int(raw.get("dp", 1)),
            tp=int(raw.get("tp", 1)),
            pp=int(raw.get("pp", 1)),
            vllm_args=normalize_arg_list(raw.get("vllm_args")),
            guidellm_args=normalize_arg_list(raw.get("guidellm_args")),
            guidellm_command=normalize_arg_list(raw.get("guidellm_command")),
            metadata=raw.get("metadata", {}),
        )
        experiments.append(experiment)

    return gpu_counts, experiments


def build_vllm_command(experiment: Experiment, gpu_count: int):
    command = [sys.executable, "-m", "vllm", "serve", experiment.model]

    if experiment.tp and not any(arg.startswith("--tensor-parallel-size") for arg in experiment.vllm_args):
        command += ["--tensor-parallel-size", str(experiment.tp)]

    if experiment.ep > 1 and "--enable-expert-parallel" not in experiment.vllm_args:
        command.append("--enable-expert-parallel")

    command += experiment.vllm_args
    return command


def vllm_port(experiment: Experiment, default: int = 8000) -> int:
    """Port the manifest asked vLLM to serve on, so readiness polls the right place."""
    args = experiment.vllm_args
    for index, arg in enumerate(args):
        if arg == "--port" and index + 1 < len(args):
            return int(args[index + 1])
        if arg.startswith("--port="):
            return int(arg.split("=", 1)[1])
    return default


def server_is_ready(port: int, timeout: float = 2.0) -> bool:
    url = f"http://localhost:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def build_guidellm_command(experiment: Experiment):
    if experiment.guidellm_command:
        return [sys.executable, "-m", "guidellm"] + experiment.guidellm_command

    command = [
        sys.executable,
        "-m",
        "guidellm",
        "run",
        "--name",
        experiment.name,
        "--model",
        experiment.model,
        "--prompt",
        experiment.prompt,
        "--batch-size",
        str(experiment.batch_size),
        "--sequence-length",
        str(experiment.sequence_length),
        "--num-steps",
        str(experiment.num_steps),
    ]
    command += experiment.guidellm_args
    return command


async def start_vllm_server(experiment: Experiment, gpu_count: int, startup_timeout: float = 900.0):
    """Launch vLLM with the manifest's parameters and wait until it serves traffic.

    Loading a large checkpoint can take many minutes on a cold HF cache, so this
    polls /health until ready rather than assuming a fixed startup delay.
    """
    console.rule(f"Starting vLLM server: {experiment.name} on {gpu_count} GPUs")
    command = build_vllm_command(experiment, gpu_count)
    port = vllm_port(experiment)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))

    console.log(f"[blue]vLLM command:[/blue] {shlex.join(command)}")

    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    def failure(stderr: str):
        return {
            "name": experiment.name,
            "engine": "vllm",
            "gpu_count": gpu_count,
            "port": port,
            "success": False,
            "stdout": "",
            "stderr": stderr,
        }, None

    deadline = asyncio.get_running_loop().time() + startup_timeout
    while True:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            console.log(f"[red]vLLM server exited during startup for {experiment.name}[/red]")
            console.log(stderr)
            return failure(stderr)

        if await asyncio.to_thread(server_is_ready, port):
            break

        if asyncio.get_running_loop().time() >= deadline:
            console.log(
                f"[red]vLLM server did not become ready within {startup_timeout:.0f}s "
                f"for {experiment.name}[/red]"
            )
            process.terminate()
            return failure(f"Timed out waiting for http://localhost:{port}/health")

        await asyncio.sleep(3)

    console.log(f"[green]vLLM server ready for {experiment.name} on port {port}[/green]")
    return {
        "name": experiment.name,
        "engine": "vllm",
        "gpu_count": gpu_count,
        "port": port,
        "success": True,
        "stdout": "",
        "stderr": "",
    }, process


def run_guidellm(experiment: Experiment, reports_dir=None, exp_id=None, timestamp=None):
    """Run guidellm, directing its JSON report into the reports directory.

    A manifest that already specifies `--output kind=json,path=...` is left
    alone -- an explicit path in the config beats our default.
    """
    console.rule(f"Running guidellm: {experiment.name}")
    command = build_guidellm_command(experiment)

    report_path = guidellm_output_path(experiment)
    if report_path is None and reports_dir is not None:
        exp_id = exp_id or experiment.name
        timestamp = timestamp or utc_timestamp()
        report_path = str(Path(reports_dir) / f"{timestamp}_{exp_id}_guidellm.json")
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        command += ["--output", f"kind=json,path={report_path}"]

    process = subprocess.run(command, capture_output=True, text=True)
    success = process.returncode == 0
    if not success:
        console.log(f"[red]guidellm failed for {experiment.name}[/red]")
        console.log(process.stderr)
    return {
        "name": experiment.name,
        "engine": "guidellm",
        "success": success,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "guidellm_output_path": report_path,
    }


def parse_latency_from_output(output: str):
    match = re.search(r"latency[:=]\s*([0-9]+(?:\.[0-9]+)?)", output, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


async def plot_results_async(results: list[dict], output_dir: Path):
    if go is None:
        console.log("[yellow]Plotly is not installed; skipping HTML result plotting.[/yellow]")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"results_{len(results)}.html"

    fig = go.Figure()
    for result in results:
        latency = parse_latency_from_output(result.get("stdout", ""))
        score = result.get("metadata", {}).get("score")
        label = f"{result['name']}:{result['engine']}"
        if latency is not None:
            fig.add_trace(go.Bar(name=label, x=[label], y=[latency], marker_color="#1f77b4"))
        elif score is not None:
            fig.add_trace(go.Bar(name=label, x=[label], y=[score], marker_color="#ff7f0e"))
        else:
            fig.add_trace(go.Bar(name=label, x=[label], y=[0], marker_color="#2ca02c"))

    fig.update_layout(
        title="Experiment Results",
        xaxis_title="Experiment",
        yaxis_title="Metric",
        barmode="group",
        template="plotly_dark",
    )

    await asyncio.to_thread(fig.write_html, str(output_file), include_plotlyjs="cdn")
    console.log(f"[green]Saved plot to {output_file}[/green]")
    return output_file


def render_runtime_banner(model: str | None = None, version: str | None = None) -> str:
    """Startup artwork. Delegates to the block-font renderer in banner.py."""
    return harness_banner(version or __version__, model)


async def run_all_experiments(
    config_path: Path,
    install: bool = True,
    output_dir: Path | None = None,
    hf_home: Path | None = None,
    experiments_dir: Path | None = None,
    reports_dir: Path | None = None,
):
    output_dir = Path(output_dir) if output_dir is not None else Path.cwd()

    # See cmd_check: the banner carries its own ANSI colour, so bypass rich.
    print(render_runtime_banner())

    if not config_path.exists():
        raise FileNotFoundError(
            f"No experiment manifest at {config_path}. Run 'rex init' to write a starter manifest."
        )

    gpu_count, _nvlink, _topo, _hf_home, experiments_dir, reports_dir = run_preflight(
        install=install,
        hf_home=hf_home,
        experiments_dir=experiments_dir if experiments_dir is not None else output_dir / DEFAULT_EXPERIMENTS_DIR,
        reports_dir=reports_dir,
    )

    gpu_counts, experiments = load_experiments(config_path)
    results = []
    pending_plots = []

    desired_gpu_counts = [count for count in gpu_counts if count <= gpu_count]
    if not desired_gpu_counts:
        raise ValueError(f"No supported GPU counts found for this machine. Detected {gpu_count}.")

    for exp in experiments:
        for count in desired_gpu_counts:
            exp_id = experiment_id(exp, count)
            timestamp = utc_timestamp()
            vllm_result, server_process = await start_vllm_server(exp, count)
            results.append(vllm_result)

            if vllm_result["success"]:
                guidellm_result = run_guidellm(exp, reports_dir, exp_id, timestamp)
                results.append(guidellm_result)
            else:
                guidellm_result = {
                    "name": exp.name,
                    "engine": "guidellm",
                    "success": False,
                    "stdout": "",
                    "stderr": "vLLM server failed to start.",
                }
                results.append(guidellm_result)

            guidellm_result["gpu_count"] = count
            save_experiment_result(guidellm_result, experiments_dir, exp_id, timestamp)

            if server_process is not None and server_process.poll() is None:
                server_process.terminate()
                try:
                    server_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server_process.kill()

            task = asyncio.create_task(plot_results_async(results.copy(), output_dir / "plots"))
            pending_plots.append(task)

    if pending_plots:
        await asyncio.gather(*pending_plots)

    output_json = output_dir / "experiment_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    console.log(f"[green]Saved results to {output_json}[/green]")
    return results
