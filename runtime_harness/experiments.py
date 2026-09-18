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


def run_stamp() -> str:
    """Unique per-run directory stamp: <utc second>-<host>-<random>.

    The bare second-resolution timestamp collided when parallel nodes ran the
    same manifest in lockstep on shared storage: two pods entered an arm in
    the same second and interleaved rocprof traces into one run directory,
    silently corrupting both copies. The host tag also records which node
    produced the run, which is what repeat-counting needs anyway.
    """
    import secrets
    import socket

    host = re.sub(r"[^A-Za-z0-9-]", "-", socket.gethostname() or "host")[:24]
    return f"{utc_timestamp()}-{host}-{secrets.token_hex(2)}"


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
    # Extra environment for the SERVER process only (e.g. RCCL_DEBUG=INFO for
    # the one-run-per-model algorithm/protocol log). Manifest values win over
    # the inherited shell so an arm's env is what the manifest says it is.
    env: dict = dataclasses.field(default_factory=dict)
    # Multiple guidellm runs against ONE server launch (a concurrency ladder
    # for a Pareto sweep, say): the model loads once and every pass reuses it.
    # Each entry is {"name", "command", "metadata"}; when non-empty,
    # ``guidellm_command`` is ignored. Passes should NOT bake an --output
    # path: the harness assigns each pass its own report and result file.
    guidellm_passes: list = dataclasses.field(default_factory=list)
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
        # A variant may add server env (RCCL_DEBUG=INFO on one arm, say)
        # without restating the base's.
        entry["env"] = {**(raw.get("env") or {}), **(variant.get("env") or {})}
        # Record what actually varied, so a result can be read back to its arm
        # without re-deriving it from the flag list.
        entry["metadata"] = {
            **raw.get("metadata", {}),
            **variant.get("metadata", {}),
            "variant": suffix,
        }
        expanded.append(entry)
    return expanded


def parse_guidellm_passes(raw):
    """Normalize a manifest's ``guidellm_passes`` list.

    Each pass must carry a name (it suffixes the report and result filenames,
    so two passes with one name would overwrite each other) and its own
    guidellm command; metadata is merged into the pass's saved result.
    """
    if not raw:
        return []
    passes = []
    seen = set()
    for item in raw:
        if "name" not in item or "guidellm_command" not in item:
            raise ValueError("Every guidellm pass needs a name and a guidellm_command.")
        name = str(item["name"])
        if name in seen:
            raise ValueError(f"Duplicate guidellm pass name {name!r}.")
        seen.add(name)
        passes.append(
            {
                "name": name,
                "command": normalize_arg_list(item["guidellm_command"]),
                "metadata": item.get("metadata") or {},
            }
        )
    return passes


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
            env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
            guidellm_passes=parse_guidellm_passes(raw.get("guidellm_passes")),
            capture=capture_spec,
        )
        experiments.append(experiment)

    return gpu_counts, experiments




def vllm_launcher() -> list[str]:
    """`python -m vllm` fails (no __main__); prefer the console script, then the
    CLI module it points at."""
    vllm_bin = shutil.which("vllm")
    if vllm_bin:
        return [vllm_bin]
    return [sys.executable, "-m", "vllm.entrypoints.cli.main"]


def build_vllm_command(experiment: Experiment, gpu_count: int):
    command = vllm_launcher() + ["serve", experiment.model]

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


def guidellm_launcher() -> list[str]:
    guidellm_bin = shutil.which("guidellm")
    if guidellm_bin:
        return [guidellm_bin]
    return [sys.executable, "-m", "guidellm"]


def parallelism_label(experiment: "Experiment") -> str:
    """Human-readable parallelism, e.g. 'TP4', 'TP2xDP4', 'TP4xDP2+EP'.

    Parsed from the actual vllm_args so it reflects what was launched, not just
    the tp field. Used in the InferenceX-style results table.
    """
    args = experiment.vllm_args

    def flag_val(flag):
        for i, a in enumerate(args):
            if a == flag and i + 1 < len(args):
                return args[i + 1]
            if a.startswith(flag + "="):
                return a.split("=", 1)[1]
        return None

    tp = flag_val("--tensor-parallel-size") or str(experiment.tp)
    dp = flag_val("--data-parallel-size")
    label = f"TP{tp}"
    if dp and dp not in ("1", None):
        label += f"xDP{dp}"
    if "--enable-expert-parallel" in args:
        label += "+EP"
    return label


#: Root-cause signatures scanned in a failed server/guidellm log, most specific
#: first. Each maps a substring to a concise reason; None means "quote the
#: matching line verbatim" (for messages that already carry the specifics).
_FAILURE_SIGNATURES = [
    ("trust_remote_code", "tokenizer needs trust_remote_code — this image's transformers does not know the model type (use a newer image)"),
    ("timed out waiting for engine core", "engine core startup exceeded VLLM_ENGINE_READY_TIMEOUT_S — raise it (DP spins up many cores)"),
    ("out of memory", "GPU out of memory — lower --gpu-memory-utilization / --max-num-seqs, or increase tensor-parallel size"),
    ("hip error", "HIP runtime error — see server.log (often OOM or an unsupported kernel/config)"),
    ("unsupported speculative method", None),
    ("no such option", None),
    ("no gpus visible in the kfd topology", "container has no GPUs — check /dev/kfd and /dev/dri device access"),
]


def extract_failure_reason(text: str) -> str:
    """One concrete sentence for why a launch failed, mined from its log.

    Prefers a known signature; otherwise returns the last exception line in the
    traceback (which is normally the root cause). Never a bare 'it failed'.
    """
    if not text or not text.strip():
        return "no log captured (server produced no output before dying)"
    low = text.lower()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for needle, reason in _FAILURE_SIGNATURES:
        if needle in low:
            if reason is not None:
                return reason
            for ln in reversed(lines):  # quote the specific line
                if needle in ln.lower():
                    return ln[:300]
    for ln in reversed(lines):  # last "SomethingError: message" line
        if re.match(r"^[\w.]*(Error|Exception|Timeout)\b.*", ln):
            return ln[:300]
    return lines[-1][:300]


def build_guidellm_command(experiment: Experiment):
    if experiment.guidellm_command:
        return guidellm_launcher() + experiment.guidellm_command

    command = [
        *guidellm_launcher(),
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
    """VRAM in use across all visible GPUs, or None if no SMI can answer.

    nvidia-smi first; on a ROCm host the same question goes to amd-smi. The
    teardown gate this feeds is vendor-neutral: an orphan holding the KV cache
    kills the next arm the same way on both."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        from . import rocm

        return rocm.gpu_memory_used_mib() if rocm.is_rocm() else None
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


async def start_vllm_server(
    experiment: Experiment,
    gpu_count: int,
    startup_timeout: float = 900.0,
    telemetry_dir: Path | None = None,
):
    """Launch vLLM with the manifest's parameters and wait until it serves traffic.

    Loading a large checkpoint can take many minutes on a cold HF cache, so this
    polls /health until ready rather than assuming a fixed startup delay.

    On a ROCm host, and when ``telemetry_dir`` is given, the server is launched
    under rocprofv3 (kernel/HIP/memcpy/RCCL planes into that directory) and its
    stderr goes to ``server.log`` there — RCCL_DEBUG output only exists on
    stderr, and a PIPE that is read only on failure would discard it exactly on
    the successful run it was collected for.
    """
    from . import rocm

    console.rule(f"Starting vLLM server: {experiment.name} on {gpu_count} GPUs")
    command = build_vllm_command(experiment, gpu_count)
    port = vllm_port(experiment)
    env = os.environ.copy()
    visible = ",".join(str(i) for i in range(gpu_count))
    # CUDA_VISIBLE_DEVICES for NVIDIA; ROCR/HIP for ROCm. Setting all three is
    # harmless on either vendor and keeps this one code path.
    env["CUDA_VISIBLE_DEVICES"] = visible
    env["ROCR_VISIBLE_DEVICES"] = visible
    env["HIP_VISIBLE_DEVICES"] = visible
    env.update(experiment.env)

    on_rocm = rocm.is_rocm()
    server_log = None
    if on_rocm and telemetry_dir is not None:
        # Kernel tracing is OFF by default: the rocprofv3 CSV/pftrace files are
        # large, and rocprofv3 has stalled long-lived servers to death. Opt IN
        # per experiment by setting REX_ROCPROF=1 in its env (e.g. one short
        # arm dedicated to capturing a trace). server.log is always kept — it
        # is how a failed launch is diagnosed — tracing or not.
        trace_on = env.get(rocm.ENV_ROCPROF, "").strip() not in ("", "0")
        if trace_on:
            prefix = rocm.rocprofv3_prefix(Path(telemetry_dir) / "rocprof")
            if prefix:
                command = prefix + command
        else:
            console.log(
                f"[dim]{experiment.name}: kernel tracing off "
                f"(set {rocm.ENV_ROCPROF}=1 to capture)[/dim]"
            )
        telemetry_dir = Path(telemetry_dir)
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        server_log = (telemetry_dir / "server.log").open("w", encoding="utf-8")

    console.log(f"[blue]vLLM command:[/blue] {shlex.join(command)}")

    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=server_log if server_log is not None else subprocess.PIPE,
        text=True,
        env=env,
        # Own process group, so stop_vllm_server can signal the whole tree.
        # Without it only the API server is reachable and EngineCore is orphaned
        # still holding the KV cache.
        start_new_session=True,
    )
    server_log_path = None
    if server_log is not None:
        # The child holds its own descriptor; the parent's copy would only leak.
        server_log_path = Path(server_log.name)
        server_log.close()

    def read_stderr() -> str:
        if server_log_path is not None:
            try:
                return server_log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
            except OSError:
                return ""
        return process.stderr.read() if process.stderr else ""

    def failure(stderr: str):
        return {
            "name": experiment.name,
            "engine": "vllm",
            "gpu_count": gpu_count,
            "port": port,
            "success": False,
            "stdout": "",
            "stderr": stderr,
            "failure_reason": extract_failure_reason(stderr),
            "server_log": str(server_log_path) if server_log_path else None,
        }, None

    deadline = asyncio.get_running_loop().time() + startup_timeout
    while True:
        if process.poll() is not None:
            stderr = read_stderr()
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

    from . import rocm

    on_rocm = rocm.is_rocm()

    for exp in experiments:
        for count in desired_gpu_counts:
            exp_id = experiment_id(exp, count)
            timestamp = run_stamp()

            if exp.capture is not None:
                if on_rocm:
                    # The gitm capture path is the CUDA injection collector;
                    # this window's ROCm kernel plane is rocprofv3 on the
                    # serving runs. Skipping loudly beats failing quietly.
                    console.log(
                        f"[yellow]{exp.name}: capture experiments are NVIDIA-only "
                        f"(gitm/CUPTI); skipped on this ROCm host.[/yellow]"
                    )
                    results.append({
                        "name": exp_id, "engine": "gitm-capture",
                        "success": False, "status": "skipped",
                        "stderr": "capture is NVIDIA-only; not run on ROCm",
                    })
                    continue
                results.extend(run_capture_experiment(exp, count, experiments_dir, timestamp))
                continue

            # Telemetry rides in the run's experiment directory, next to the
            # guidellm result it belongs to. Async and cheap by construction;
            # the serving numbers are the thing it must never perturb.
            telemetry_dir = Path(experiments_dir) / exp_id / f"{timestamp}_{exp_id}_telemetry"
            sampler = None
            if on_rocm:
                telemetry_dir.mkdir(parents=True, exist_ok=True)
                (telemetry_dir / "nic_before.json").write_text(
                    json.dumps(rocm.nic_counters(), indent=2), encoding="utf-8")
                sampler = rocm.AmdSmiSampler(telemetry_dir / "amdsmi_samples.jsonl").start()

            try:
                vllm_result, server_process = await start_vllm_server(
                    exp, count, telemetry_dir=telemetry_dir if on_rocm else None
                )
                results.append(vllm_result)

                if vllm_result["success"] and exp.guidellm_passes:
                    # One server, many traffic points (a Pareto concurrency
                    # ladder): the model loads once and every pass reuses it.
                    # Each pass gets its own report and result file; a failed
                    # pass is recorded and the remaining passes still run.
                    for gpass in exp.guidellm_passes:
                        if not server_is_ready(vllm_port(exp)):
                            console.log(
                                f"[red]vLLM server no longer serving before pass "
                                f"{gpass['name']}; recording remaining passes as failed[/red]"
                            )
                            guidellm_result = {
                                "name": f"{exp.name}-{gpass['name']}",
                                "engine": "guidellm",
                                "success": False,
                                "stdout": "",
                                "stderr": "vLLM server died mid-sweep; pass not run.",
                                "failure_reason": vllm_result.get("failure_reason")
                                or "vLLM server died mid-sweep",
                                "server_log": vllm_result.get("server_log"),
                                "gpu_count": count,
                                "pass": gpass["name"],
                                "parallelism": parallelism_label(exp),
                                "precision": exp.metadata.get("precision", "INT4"),
                                "metadata": {**exp.metadata, **gpass["metadata"]},
                            }
                            results.append(guidellm_result)
                            save_experiment_result(
                                guidellm_result, experiments_dir, exp_id,
                                f"{timestamp}_{gpass['name']}",
                            )
                            continue
                        pass_exp = dataclasses.replace(
                            exp,
                            name=f"{exp.name}-{gpass['name']}",
                            guidellm_command=list(gpass["command"]),
                        )
                        guidellm_result = run_guidellm(
                            pass_exp, reports_dir, f"{exp_id}-{gpass['name']}", timestamp
                        )
                        guidellm_result["gpu_count"] = count
                        guidellm_result["pass"] = gpass["name"]
                        guidellm_result["parallelism"] = parallelism_label(exp)
                        guidellm_result["precision"] = exp.metadata.get("precision", "INT4")
                        guidellm_result["metadata"] = {
                            **exp.metadata,
                            **gpass["metadata"],
                        }
                        if not guidellm_result.get("success"):
                            guidellm_result["failure_reason"] = extract_failure_reason(
                                guidellm_result.get("stderr", "")
                            )
                        if on_rocm:
                            guidellm_result["telemetry_dir"] = str(telemetry_dir)
                        results.append(guidellm_result)
                        save_experiment_result(
                            guidellm_result,
                            experiments_dir,
                            exp_id,
                            f"{timestamp}_{gpass['name']}",
                        )
                else:
                    if vllm_result["success"]:
                        guidellm_result = run_guidellm(exp, reports_dir, exp_id, timestamp)
                        if not guidellm_result.get("success"):
                            guidellm_result["failure_reason"] = extract_failure_reason(
                                guidellm_result.get("stderr", "")
                            )
                        results.append(guidellm_result)
                    else:
                        guidellm_result = {
                            "name": exp.name,
                            "engine": "guidellm",
                            "success": False,
                            "stdout": "",
                            "stderr": "vLLM server failed to start.",
                            "failure_reason": vllm_result.get("failure_reason")
                            or "vLLM server failed to start (no reason captured)",
                            "server_log": vllm_result.get("server_log"),
                        }
                        results.append(guidellm_result)

                    guidellm_result["gpu_count"] = count
                    guidellm_result["parallelism"] = parallelism_label(exp)
                    guidellm_result["precision"] = exp.metadata.get("precision", "INT4")
                    if on_rocm:
                        guidellm_result["telemetry_dir"] = str(telemetry_dir)
                    save_experiment_result(guidellm_result, experiments_dir, exp_id, timestamp)

                if server_process is not None:
                    stop_vllm_server(server_process)
            finally:
                if sampler is not None:
                    sampler.stop()
                if on_rocm:
                    (telemetry_dir / "nic_after.json").write_text(
                        json.dumps(rocm.nic_counters(), indent=2), encoding="utf-8")

            task = asyncio.create_task(plot_results_async(results.copy(), output_dir / "plots"))
            pending_plots.append(task)

    if pending_plots:
        await asyncio.gather(*pending_plots)

    # More than one tracing arm in the results means an overhead comparison was
    # the point of the sweep; print it rather than leaving it in the JSON.
    capture_records = [r for r in results if r.get("engine") == "gitm-capture"]
    if len({(r.get("serving") or {}).get("tracing") or r.get("arm") for r in capture_records}) > 1:
        console.print(render_overhead(overhead_rows(capture_records)))

    # Every serving run ends with the InferenceX-standard metrics table:
    # E2E / TTFT / ITL tails plus tok/s/user and tok/s/GPU, one row per
    # (experiment, pass). Reads each guidellm report from disk, so it is
    # accurate even when a pass failed or the server died mid-sweep.
    from .serving_metrics import print_serving_metrics

    print_serving_metrics(results)

    output_json = output_dir / "experiment_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    console.log(f"[green]Saved results to {output_json}[/green]")
    return results
