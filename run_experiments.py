#!/usr/bin/env python3
import argparse
import asyncio
import dataclasses
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import plotly.graph_objects as go
import yaml
from rich.console import Console

console = Console()

PROJECT_ROOT = Path(__file__).parent
DEFAULT_CONFIG = PROJECT_ROOT / "experiments.yaml"


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


def ensure_dependencies():
    console.rule("Installing runtime dependencies")
    deps = ["vllm", "guidellm", "plotly", "pyyaml", "rich"]
    command = f"{sys.executable} -m pip install {' '.join(deps)}"
    run_command(command)
    console.log("[green]Dependencies installed successfully[/green]")


def parse_gpu_topology():
    if shutil.which("nvidia-smi") is None:
        raise FileNotFoundError("nvidia-smi not found in PATH. NVIDIA drivers must be installed.")

    topo_raw = run_command("nvidia-smi topo --matrix", capture_output=True)
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


async def start_vllm_server(experiment: Experiment, gpu_count: int):
    console.rule(f"Starting vLLM server: {experiment.name} on {gpu_count} GPUs")
    command = build_vllm_command(experiment, gpu_count)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))

    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    await asyncio.sleep(6)
    if process.poll() is not None:
        stderr = process.stderr.read() if process.stderr else ""
        console.log(f"[red]vLLM server failed to start for {experiment.name}[/red]")
        console.log(stderr)
        return {
            "name": experiment.name,
            "engine": "vllm",
            "gpu_count": gpu_count,
            "success": False,
            "stdout": "",
            "stderr": stderr,
        }, None

    console.log(f"[green]vLLM server started for {experiment.name}[/green]")
    return {
        "name": experiment.name,
        "engine": "vllm",
        "gpu_count": gpu_count,
        "success": True,
        "stdout": "",
        "stderr": "",
    }, process


def run_guidellm(experiment: Experiment):
    console.rule(f"Running guidellm: {experiment.name}")
    command = build_guidellm_command(experiment)
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
    }


def parse_latency_from_output(output: str):
    match = re.search(r"latency[:=]\s*([0-9]+(?:\.[0-9]+)?)", output, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


async def plot_results_async(results: list[dict], output_dir: Path):
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


async def run_all_experiments(config_path: Path, install: bool):
    if install:
        ensure_dependencies()

    gpu_count, nvlink_available, topo_raw = parse_gpu_topology()
    console.print(f"[bold]Detected GPUs:[/bold] {gpu_count}")
    console.print(f"[bold]NVLink available:[/bold] {nvlink_available}")
    console.print("[bold]Topology matrix:[/bold]\n" + topo_raw)

    gpu_counts, experiments = load_experiments(config_path)
    results = []
    pending_plots = []

    desired_gpu_counts = [count for count in gpu_counts if count <= gpu_count]
    if not desired_gpu_counts:
        raise ValueError(f"No supported GPU counts found for this machine. Detected {gpu_count}.")

    for exp in experiments:
        for count in desired_gpu_counts:
            vllm_result, server_process = await start_vllm_server(exp, count)
            results.append(vllm_result)

            if vllm_result["success"]:
                guidellm_result = run_guidellm(exp)
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

            if server_process is not None and server_process.poll() is None:
                server_process.terminate()
                try:
                    server_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server_process.kill()

            task = asyncio.create_task(plot_results_async(results.copy(), PROJECT_ROOT / "plots"))
            pending_plots.append(task)

    if pending_plots:
        await asyncio.gather(*pending_plots)

    output_json = PROJECT_ROOT / "experiment_results.json"
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    console.log(f"[green]Saved results to {output_json}[/green]")


def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM / guidellm experiments on H100 NVLink clusters.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to experiments YAML manifest.")
    parser.add_argument("--install", action="store_true", help="Install vllm, guidellm, and plotting dependencies before running.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run_all_experiments(args.config, args.install))
    except Exception as exc:
        console.print(f"[bold red]Experiment harness failed:[/bold red] {exc}")
        sys.exit(1)



def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM / guidellm experiments on H100 NVLink clusters.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to experiments YAML manifest.")
    parser.add_argument("--install", action="store_true", help="Install vllm, guidellm, and plotting dependencies before running.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run_all_experiments(args.config, args.install))
    except Exception as exc:
        console.print(f"[bold red]Experiment harness failed:[/bold red] {exc}")
        sys.exit(1)
