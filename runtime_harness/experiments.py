"""Experiment orchestration: launch vLLM, drive guidellm, collect results."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
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
from .analysis import analyze_capture, overhead_rows, print_capture_analysis, render_overhead
from .banner import harness_banner
from .capture import (
    ARM_TRACING,
    CaptureSpec,
    capture_spec_from_raw,
    check_keep_server,
    run_capture,
    warn_if_no_hf_token,
)
from .checks import DEFAULT_EXPERIMENTS_DIR, run_preflight

console = Console()

# Config and outputs resolve against the invocation directory, never the install
# location -- site-packages is often read-only and is never the user's workspace.
DEFAULT_CONFIG_NAME = "experiments.yaml"
DATA_DIR = Path(__file__).parent / "data"
BUNDLED_CONFIG = DATA_DIR / DEFAULT_CONFIG_NAME

# Starter manifests shipped in the wheel, written out by `rex init --template`.
# One copy each, here -- a second copy in the repo root would drift, and a sweep
# whose arms differ in more than the thing under test measures nothing.
TEMPLATES = {
    # guidellm: end-to-end serving throughput.
    "serving": BUNDLED_CONFIG,
    "cudagraph-spec": DATA_DIR / "cudagraph-spec.yaml",
    # gitm capture: kernels, NVTX layer/op attribution, prefill vs decode.
    "capture": DATA_DIR / "capture.yaml",
    "overhead": DATA_DIR / "overhead.yaml",
    "cudagraph-spec-capture": DATA_DIR / "cudagraph-spec-capture.yaml",
}


def template_path(name: str) -> Path:
    try:
        return TEMPLATES[name]
    except KeyError:
        raise ValueError(
            f"Unknown template {name!r}. Available: {', '.join(sorted(TEMPLATES))}."
        ) from None


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
    # Present when the entry is driven by `gitm capture serve` instead of by
    # guidellm. The two are exclusive: gitm owns the server for a capture, because
    # the CUDA driver reads CUDA_INJECTION64_PATH once at CUDA init and a server
    # this harness started itself can never be traced afterwards.
    capture: CaptureSpec | None = None


def normalize_arg_list(raw):
    if raw is None:
        return []
    if isinstance(raw, str):
        return shlex.split(raw)
    if isinstance(raw, list):
        return [str(item) for item in raw]
    raise ValueError("Argument lists must be strings or lists.")


def retarget_output_path(command: list[str], suffix: str) -> list[str]:
    """Give one variant its own guidellm report path.

    The path is baked into a ``--output kind=json,path=...`` argument. Left alone,
    every arm of a sweep writes the same file and the last one wins silently -- the
    run directory still looks complete, and the JSON in it describes whichever arm
    happened to finish last.
    """
    out = []
    for arg in command:
        match = re.search(r"(kind=json,\s*path=)([^,\s]+)", arg)
        if match:
            path = Path(match.group(2))
            renamed = path.with_name(f"{path.stem}-{suffix}{path.suffix}")
            arg = arg[: match.start(2)] + str(renamed) + arg[match.end(2) :]
        out.append(arg)
    return out


def expand_variants(raw: dict) -> list[dict]:
    """Expand one entry carrying ``variants:`` into one entry per variant.

    A sweep over cudagraph modes or speculative settings is the same experiment
    with a few flags changed. Writing those out by hand means N near-identical
    blocks that drift apart on the fields nobody meant to vary -- and a sweep whose
    arms differ in more than the one thing under test measures nothing.

    Each variant contributes a name suffix and its own ``vllm_args``, appended
    AFTER the base list so a variant can override a base flag (vLLM's parser takes
    the last occurrence). Its guidellm report is retargeted so arms do not
    overwrite each other.
    """
    variants = raw.get("variants")
    if not variants:
        return [raw]

    expanded = []
    for variant in variants:
        if "name" not in variant:
            raise ValueError(f"Every variant of {raw.get('name')!r} needs a name.")
        suffix = str(variant["name"])
        entry = {key: value for key, value in raw.items() if key != "variants"}
        entry["name"] = f"{raw['name']}-{suffix}"
        entry["vllm_args"] = normalize_arg_list(raw.get("vllm_args")) + normalize_arg_list(
            variant.get("vllm_args")
        )
        entry["guidellm_command"] = retarget_output_path(
            normalize_arg_list(raw.get("guidellm_command")), suffix
        )
        # A variant may override individual capture settings (a longer window for
        # the slow arm, say) without restating the whole block -- and without the
        # load shape silently differing between arms, which is the one thing a
        # sweep cannot survive.
        base_capture = raw.get("capture")
        variant_capture = variant.get("capture")
        if isinstance(base_capture, dict) or isinstance(variant_capture, dict):
            entry["capture"] = {
                **(base_capture if isinstance(base_capture, dict) else {}),
                **(variant_capture if isinstance(variant_capture, dict) else {}),
            }
        elif variant_capture is not None:
            entry["capture"] = variant_capture
        # Record what actually varied, so a result can be read back to its arm
        # without re-deriving it from the flag list.
        entry["metadata"] = {
            **raw.get("metadata", {}),
            **variant.get("metadata", {}),
            "variant": suffix,
        }
        expanded.append(entry)
    return expanded


def load_experiments(config_path: Path):
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    gpu_counts = config.get("gpu_counts", [2, 4, 8])
    experiments = []
    raw_entries = [
        expanded
        for entry in config.get("experiments", [])
        for expanded in expand_variants(entry)
    ]
    for raw in raw_entries:
        capture_spec = capture_spec_from_raw(raw.get("capture"))
        # guidellm needs a prompt; gitm builds its own synthetic ones from
        # --input-tokens, so requiring one there would be a field nobody reads.
        if capture_spec is None and "prompt" not in raw:
            raise ValueError(f"Experiment {raw.get('name')!r} needs a prompt (or a capture block).")
        experiment = Experiment(
            name=raw["name"],
            model=raw["model"],
            prompt=raw.get("prompt", ""),
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
            capture=capture_spec,
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


def gpu_memory_used_mib():
    """VRAM in use across all visible GPUs, or None if nvidia-smi cannot answer."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    total = 0
    for line in proc.stdout.split():
        try:
            total += int(line.strip())
        except ValueError:
            return None
    return total


def stop_vllm_server(process, *, idle_mib: int = 2048, settle_timeout: float = 180.0):
    """Kill the server's whole process GROUP, then wait for its VRAM to come back.

    vLLM V1 is a process tree -- the API server, EngineCore, and one worker per TP
    rank -- and EngineCore is the one holding the KV cache, which at
    --gpu-memory-utilization 0.95 is ~133 GiB on an H200. ``process.terminate()``
    and ``process.kill()`` signal only the direct child, so EngineCore survives.
    SIGKILL is worse than SIGTERM here rather than better: it denies the parent any
    chance to reap its own children, which makes the orphan more likely, not less.

    The orphan does not hold the port. So the next experiment's /health poll passes,
    any port check passes, and it fails instead inside engine startup with
    "Engine core initialization failed ... Failed core proc(s): {}" -- a message that
    mentions neither memory nor the previous run. Unattended, every arm after the
    first fails that way and is recorded as success=False, leaving a results file
    that looks complete and holds one real measurement.

    Hence both halves: signal the group (start_new_session in start_vllm_server puts
    the tree in its own group), and then verify the memory actually came back rather
    than assuming a fixed sleep was long enough. Freeing lags process exit.
    """
    if process is None or process.poll() is not None:
        return
    try:
        pgid = os.getpgid(process.pid)
    except (OSError, AttributeError):
        pgid = None

    for sig, grace in ((signal.SIGINT, 20.0), (signal.SIGTERM, 15.0), (signal.SIGKILL, 5.0)):
        if process.poll() is not None:
            break
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:  # no process group -- the direct child is all we can reach
                process.send_signal(sig)
        except (ProcessLookupError, PermissionError, OSError):
            break
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue

    wait_for_idle_vram(idle_mib=idle_mib, settle_timeout=settle_timeout)


def wait_for_idle_vram(idle_mib: int = 2048, settle_timeout: float = 180.0) -> bool:
    """Block until VRAM is back below ``idle_mib``, or report that it never was.

    Freeing lags process exit, so "the server exited" is not "the memory is back".
    Used after every experiment -- including the captures, where gitm owns the
    shutdown -- because the failure this prevents does not look like a memory
    failure: the next run's /health poll passes against nothing, and it dies inside
    engine startup with a message naming neither memory nor the previous run.
    """
    if gpu_memory_used_mib() is None:
        console.log("[yellow]nvidia-smi unavailable; not verifying VRAM was released[/yellow]")
        return True

    deadline = time.monotonic() + settle_timeout
    while time.monotonic() < deadline:
        used = gpu_memory_used_mib()
        if used is None or used <= idle_mib:
            console.log(f"[green]VRAM released ({used} MiB in use)[/green]")
            return True
        time.sleep(3.0)

    console.log(
        f"[red]VRAM still held ({gpu_memory_used_mib()} MiB) after {settle_timeout:.0f}s. "
        f"A process outside this group is holding it; the next experiment will fail "
        f"inside engine startup. Check: nvidia-smi --query-compute-apps=pid,used_memory "
        f"--format=csv[/red]"
    )
    return False


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
        # Own process group, so stop_vllm_server can signal the whole tree.
        # Without it only the API server is reachable and EngineCore is orphaned
        # still holding the KV cache.
        start_new_session=True,
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


def capture_run_label(spec: CaptureSpec, arm: str, repetition: int) -> str:
    """Directory suffix for one (arm, repetition).

    The repetition index is in the name only when there is more than one, so a
    single-arm capture reads as `..._cupti` rather than `..._cupti-r1`.
    """
    return arm if spec.repeat == 1 else f"{arm}-r{repetition}"


def run_capture_experiment(experiment: Experiment, gpu_count: int, experiments_dir, timestamp=None):
    """Run every (arm, repetition) of one capture experiment and analyze each.

    One directory per run, all under experiments/<exp-id>/, so the arms of a
    comparison sit side by side and none of them can overwrite another's trace.
    """
    spec = experiment.capture
    exp_id = experiment_id(experiment, gpu_count)
    timestamp = timestamp or utc_timestamp()
    experiments_dir = Path(experiments_dir)

    records = []
    for arm, repetition in spec.runs():
        label = capture_run_label(spec, arm, repetition)
        out_dir = experiments_dir / exp_id / f"{timestamp}_{exp_id}_{label}"
        record = run_capture(
            spec,
            arm,
            out_dir,
            experiment.model,
            experiment.vllm_args,
            gpu_count=gpu_count,
        )
        record.update(
            {
                "name": exp_id,
                "experiment": experiment.name,
                "gpu_count": gpu_count,
                "repetition": repetition,
                "timestamp": timestamp,
                "metadata": experiment.metadata,
            }
        )

        if spec.analyze and record["status"] != "skipped":
            analysis = analyze_capture(out_dir, layerwise=(arm == "nvtx"))
            record["analysis"] = analysis
            record["serving"] = analysis.get("serving")
            print_capture_analysis(analysis)
            verify_arm(record, arm)

        # gitm shuts the server down itself, but freeing lags exit and the next arm
        # starts immediately.
        if not spec.keep_server:
            wait_for_idle_vram()

        with (out_dir / "harness_record.json").open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        records.append(record)

    return records


def verify_arm(record: dict, arm: str):
    """Check the run reported the tracing state the arm asked for.

    gitm writes what it actually did into serving_summary.json. An arm that asked
    for no collector and reports one -- an injection variable inherited from the
    shell -- still produces a throughput number, and that number reads as "tracing
    is free" rather than as a broken baseline.
    """
    expected = ARM_TRACING.get(arm)
    reported = ((record.get("serving") or {}) or {}).get("tracing")
    if expected and reported and reported != expected:
        console.log(
            f"[red]{record['name']} ran as arm {arm!r} (expecting tracing={expected!r}) "
            f"but the run reports tracing={reported!r}. Do not compare this run: the "
            f"collector was not in the state the arm claims.[/red]"
        )
        record["arm_mismatch"] = {"expected": expected, "reported": reported}


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

    # Checked across the whole manifest, before anything launches: a kept server
    # holds the port and the KV cache, so it is only ever valid as the last run.
    total_capture_runs = sum(
        len(exp.capture.runs()) * len(desired_gpu_counts)
        for exp in experiments
        if exp.capture is not None
    )
    for exp in experiments:
        if exp.capture is not None:
            check_keep_server(exp.capture, total_capture_runs)
    if any(exp.capture is not None for exp in experiments):
        warn_if_no_hf_token()

    for exp in experiments:
        for count in desired_gpu_counts:
            exp_id = experiment_id(exp, count)
            timestamp = utc_timestamp()

            if exp.capture is not None:
                results.extend(run_capture_experiment(exp, count, experiments_dir, timestamp))
                continue

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

            stop_vllm_server(server_process)

            task = asyncio.create_task(plot_results_async(results.copy(), output_dir / "plots"))
            pending_plots.append(task)

    if pending_plots:
        await asyncio.gather(*pending_plots)

    # More than one tracing arm in the results means an overhead comparison was
    # the point of the sweep; print it rather than leaving it in the JSON.
    capture_records = [r for r in results if r.get("engine") == "gitm-capture"]
    if len({(r.get("serving") or {}).get("tracing") or r.get("arm") for r in capture_records}) > 1:
        console.print(render_overhead(overhead_rows(capture_records)))

    output_json = output_dir / "experiment_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    console.log(f"[green]Saved results to {output_json}[/green]")
    return results
