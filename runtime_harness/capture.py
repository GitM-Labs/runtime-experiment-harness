"""Drive `gitm capture serve` from an experiment manifest.

The guidellm path in :mod:`runtime_harness.experiments` owns the server: it starts
`vllm serve`, drives load at it, and tears it down. The capture path does not --
`gitm capture serve` owns the whole lifecycle, because the CUDA driver reads
``CUDA_INJECTION64_PATH`` exactly once at CUDA init, so the collector has to be in
the environment *before* the server process starts. A server the harness launched
itself can never be traced afterwards.

So an experiment with a ``capture:`` block turns into one `gitm capture serve`
invocation per (arm, repetition), with the manifest's ``vllm_args`` passed through
after ``--``. What comes back is a directory of artifacts (trace.jsonl,
kernel_breakdown.json, serving_summary.json, server.log) that
:mod:`runtime_harness.analysis` reads.
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import subprocess
from pathlib import Path

from rich.console import Console

console = Console()

# How an arm name maps onto the flags that put the collector in that state.
#
#   off    the baseline of an overhead measurement. gitm *clears* the injection
#          variables rather than merely not setting them -- one left exported in
#          the shell from an earlier run would attach the collector to the arm
#          whose entire purpose is not to have one.
#   cupti  kernel collection only: CONCURRENT_KERNEL + MEMCPY + SYNCHRONIZATION.
#   nvtx   adds the correlation chain (RUNTIME + DRIVER + MARKER) *and* vLLM's
#          --enable-layerwise-nvtx-tracing. Both halves are required and neither
#          errors alone: ranges nobody collects, and collection with no ranges,
#          each produce a clean trace with range_op null on every kernel. gitm's
#          --nvtx sets both, which is why the manifest never sets GITM_TRACE_NVTX
#          or NVTX_INJECTION64_PATH by hand.
ARM_FLAGS = {
    "off": ["--no-trace"],
    "cupti": [],
    "nvtx": ["--nvtx"],
}

# What gitm records in serving_summary.json["tracing"] for each arm. Used to
# verify an arm ran in the state it claimed rather than trusting the flag we
# passed -- the whole failure mode of an overhead measurement is an arm that
# reports a throughput number from the wrong tracing state.
ARM_TRACING = {"off": "off", "cupti": "cupti", "nvtx": "cupti+nvtx"}

DEFAULT_ARMS = ["cupti"]

# Artifacts gitm writes into --out. A capture that produced none of them did not
# run; one missing only trace.jsonl is the untraced (--no-trace) arm.
CAPTURE_ARTIFACTS = (
    "trace.jsonl",
    "kernel_breakdown.json",
    "serving_summary.json",
    "run_manifest.json",
    "preflight.json",
    "server.log",
)


@dataclasses.dataclass
class CaptureSpec:
    """The ``capture:`` block of a manifest entry.

    Field names match `gitm capture serve` flags one for one. The load shape
    (requests/concurrency/input/output tokens) is what distinguishes a decode-heavy
    capture from a prefill-heavy one, and it is the thing that must stay identical
    across the arms of a comparison -- hence one spec per experiment, applied to
    every arm, rather than per-arm load settings.
    """

    requests: int = 512
    concurrency: int = 256
    input_tokens: int = 1024
    output_tokens: int = 256
    warmup: int = 8
    seed: int = 42
    arms: list[str] = dataclasses.field(default_factory=lambda: list(DEFAULT_ARMS))
    repeat: int = 1
    keep_server: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    health_timeout: float | None = None
    request_timeout: float | None = None
    no_ignore_eos: bool = False
    skip_preflight: bool = False
    dry_run: bool = False
    extra_args: list[str] = dataclasses.field(default_factory=list)
    analyze: bool = True

    def runs(self):
        """(arm, repetition) pairs in execution order, repetitions interleaved.

        Arms rotate on the inner loop -- off, cupti, nvtx, off, cupti, nvtx --
        rather than three offs then three cuptis. Anything that drifts over the
        session (thermals, another tenant on the box, a background compaction)
        biases whichever arm ran during it; interleaving spreads that across all
        three instead of donating it entirely to one.
        """
        return [(arm, rep) for rep in range(1, self.repeat + 1) for arm in self.arms]


def capture_spec_from_raw(raw):
    """Build a CaptureSpec from a manifest mapping, or None when there is none."""
    if raw is None:
        return None
    if raw is True:  # `capture: true` -- everything default
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("`capture:` must be a mapping of gitm capture settings.")

    known = {field.name for field in dataclasses.fields(CaptureSpec)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Unknown capture settings: {', '.join(sorted(unknown))}. "
            f"Known: {', '.join(sorted(known))}."
        )

    spec = CaptureSpec(**{key: value for key, value in raw.items() if key in known})
    spec.arms = [str(arm) for arm in (spec.arms or DEFAULT_ARMS)]
    spec.extra_args = [str(arg) for arg in spec.extra_args]

    bad = [arm for arm in spec.arms if arm not in ARM_FLAGS]
    if bad:
        raise ValueError(
            f"Unknown capture arm(s): {', '.join(bad)}. Known arms: {', '.join(ARM_FLAGS)}."
        )
    if spec.repeat < 1:
        raise ValueError("capture.repeat must be at least 1.")
    if not spec.arms:
        raise ValueError("capture.arms must name at least one arm.")
    return spec


def build_capture_command(spec: CaptureSpec, arm: str, out_dir: Path, model: str, vllm_args=None):
    """Assemble one `gitm capture serve ... -- vllm serve <model> ...` invocation.

    The serve command goes after ``--`` verbatim, so the manifest's vLLM flags are
    the ones that run. gitm appends --host/--port itself (it has to know where to
    send load) and adds --tensor-parallel-size when the command does not pin one.
    """
    if arm not in ARM_FLAGS:
        raise ValueError(f"Unknown capture arm {arm!r}. Known arms: {', '.join(ARM_FLAGS)}.")

    command = [
        "gitm", "capture", "serve",
        "--out", str(out_dir),
        "--requests", str(spec.requests),
        "--concurrency", str(spec.concurrency),
        "--input-tokens", str(spec.input_tokens),
        "--output-tokens", str(spec.output_tokens),
        "--warmup", str(spec.warmup),
        "--seed", str(spec.seed),
        "--host", spec.host,
        "--port", str(spec.port),
    ]
    if spec.health_timeout is not None:
        command += ["--health-timeout", str(spec.health_timeout)]
    if spec.request_timeout is not None:
        command += ["--request-timeout", str(spec.request_timeout)]
    if spec.no_ignore_eos:
        command.append("--no-ignore-eos")
    if spec.skip_preflight:
        command.append("--skip-preflight")
    if spec.dry_run:
        command.append("--dry-run")
    if spec.keep_server:
        command.append("--keep-server")

    command += ARM_FLAGS[arm]
    command += spec.extra_args
    command += ["--", "vllm", "serve", model]
    command += [str(arg) for arg in (vllm_args or [])]
    return command


def check_keep_server(spec: CaptureSpec, total_runs: int):
    """Refuse --keep-server when another run follows it.

    A kept server holds the port and the KV cache -- ~133 GiB at
    --gpu-memory-utilization 0.95 on an H200. The next arm then fails inside engine
    startup with a message that mentions neither memory nor the previous run, and
    every arm after the first is recorded as a failure while the results file still
    looks complete. Worth refusing up front rather than discovering it an hour into
    a sweep.
    """
    if spec.keep_server and total_runs > 1:
        raise ValueError(
            f"capture.keep_server leaves the server holding the port and the KV cache, "
            f"but this manifest schedules {total_runs} runs. Every run after the first "
            f"would fail inside engine startup. Set keep_server: false, or reduce the "
            f"manifest to a single arm and repetition."
        )


def warn_if_no_hf_token():
    """A gated checkpoint fails after the server has already started."""
    if not any(os.environ.get(name) for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")):
        console.log(
            "[yellow]HF_TOKEN is not set. A gated checkpoint will fail during weight "
            "download, which happens minutes into the run and after preflight has "
            "already passed. export HF_TOKEN=... before the sweep.[/yellow]"
        )


def run_capture(
    spec: CaptureSpec,
    arm: str,
    out_dir: Path,
    model: str,
    vllm_args=None,
    gpu_count=None,
    env=None,
):
    """Run one capture and report what landed on disk.

    gitm's own exit codes are preserved in the record: 0 captured, 1 failed,
    2 skipped (preflight says this box cannot run it). A skip is not a success and
    not a crash, and flattening the three into a boolean would let a sweep that
    captured nothing read as a sweep that ran.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = build_capture_command(spec, arm, out_dir, model, vllm_args)

    env = dict(os.environ if env is None else env)
    if gpu_count is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(gpu_count))

    console.rule(f"gitm capture serve: {out_dir.name} [{arm}]")
    console.log(f"[blue]capture command:[/blue] {shlex.join(command)}")

    # Streamed to the terminal and to a file rather than captured: a capture can run
    # for many minutes past the weight load, and a silent terminal for that long is
    # indistinguishable from a hang.
    log_path = out_dir / "capture.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
            )
            for line in process.stdout:
                console.out(line.rstrip())
                log.write(line)
            returncode = process.wait()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "gitm not found on PATH. Run `rex install` to clone and install the runtime."
        ) from exc

    status = {0: "captured", 1: "failed", 2: "skipped"}.get(returncode, "failed")
    record = {
        "name": out_dir.name,
        "engine": "gitm-capture",
        "arm": arm,
        "success": returncode == 0,
        "status": status,
        "returncode": returncode,
        "capture_dir": str(out_dir),
        "command": command,
        "artifacts": sorted(
            name for name in CAPTURE_ARTIFACTS if (out_dir / name).exists()
        ),
    }

    if status != "captured":
        console.log(
            f"[red]capture {status} (exit {returncode}) for {out_dir.name}; "
            f"see {log_path} and {out_dir / 'server.log'}[/red]"
        )
    return record
