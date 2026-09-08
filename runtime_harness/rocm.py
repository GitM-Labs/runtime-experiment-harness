"""ROCm backend: vendor detection, amd-smi checks, telemetry, rocprofv3.

Design constraints for MI355X/MI300X serving sweeps:

* Same experiments.yaml, guidellm traffic, metadata, and results schema as the
  NVIDIA path — these runs must be directly comparable to the H200 rows.
* No GITM capture install on ROCm this window; kernel plane is rocprofv3.
* All telemetry is async and lands in the existing experiment directory.
* NEVER pip install torch/vllm here: the vLLM ROCm image ships matched builds,
  and PyPI would replace them with CUDA wheels that cannot import.
* No hardware perf counters (--pmc) on serving runs — counter collection
  serializes kernels. That is offline Tier-2 work, not this harness.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from rich.console import Console

console = Console()

#: Escape hatch: set REX_NO_ROCPROF=1 to launch servers unwrapped (a clean arm
#: for overhead comparison, or when rocprofv3 itself is the suspect).
ENV_NO_ROCPROF = "REX_NO_ROCPROF"

#: Sampler cadence. The plan asks for 100 ms or better; the library path meets
#: it, the CLI fallback records what it actually achieved instead of lying.
SAMPLE_INTERVAL_S = 0.1
#: The full gpu-metrics table (throttle status, PCIe, xGMI accumulators) is
#: bulky; snapshot it at 1 Hz rather than every tick.
SLOW_SAMPLE_EVERY = 10


def is_rocm() -> bool:
    """True on a ROCm host. kfd is the ground truth; CLI presence is a hint."""
    if Path("/sys/class/kfd/kfd/topology/nodes").is_dir():
        return True
    return shutil.which("nvidia-smi") is None and (
        shutil.which("amd-smi") is not None or shutil.which("rocm-smi") is not None
    )


def rocm_version() -> str | None:
    info = Path("/opt/rocm/.info/version")
    if info.exists():
        return info.read_text().strip()
    if shutil.which("amd-smi"):
        try:
            out = subprocess.run(
                ["amd-smi", "version"], capture_output=True, text=True, timeout=30
            ).stdout
            match = re.search(r"ROCm version:\s*([\w.\-]+)", out)
            if match:
                return match.group(1)
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def gpu_count() -> int:
    """GPU count from kfd (nodes with a non-zero gpu_id are GPUs)."""
    nodes = Path("/sys/class/kfd/kfd/topology/nodes")
    count = 0
    if nodes.is_dir():
        for node in nodes.iterdir():
            try:
                if (node / "gpu_id").read_text().strip() not in ("", "0"):
                    count += 1
            except OSError:
                continue
    return count


def parse_gpu_topology():
    """(gpu_count, xgmi_available, raw topology text) — the NVLink probe's shape.

    xGMI plays NVLink's role on MI300X/MI355X; the raw matrix is recorded in the
    preflight output the same way the nvidia-smi matrix is.
    """
    raw = ""
    for argv in (["amd-smi", "topology"], ["rocm-smi", "--showtopo"]):
        if shutil.which(argv[0]) is None:
            continue
        try:
            raw = subprocess.run(
                argv, capture_output=True, text=True, timeout=60
            ).stdout
            if raw.strip():
                break
        except (OSError, subprocess.SubprocessError):
            continue
    xgmi = "XGMI" in raw.upper()
    return gpu_count(), xgmi, raw


def gpu_memory_used_mib():
    """VRAM in use across all GPUs, or None — amd-smi's answer to the
    nvidia-smi query that gates run-to-run teardown."""
    if shutil.which("amd-smi"):
        try:
            proc = subprocess.run(
                ["amd-smi", "metric", "--mem-usage", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0:
                total = _sum_json_field(json.loads(proc.stdout), ("used_vram", "vram_used"))
                if total is not None:
                    return total
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        # Older amd-smi spells the section --mem; try before giving up.
        try:
            proc = subprocess.run(
                ["amd-smi", "metric", "--mem", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0:
                total = _sum_json_field(json.loads(proc.stdout), ("used_vram", "vram_used"))
                if total is not None:
                    return total
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    if shutil.which("rocm-smi"):
        try:
            proc = subprocess.run(
                ["rocm-smi", "--showmemuse", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0:
                data = json.loads(proc.stdout)
                total = 0
                found = False
                for card in data.values():
                    if not isinstance(card, dict):
                        continue
                    for key, value in card.items():
                        if "VRAM" in key.upper() and "USED" in key.upper():
                            try:
                                total += int(value) // (1024 * 1024)
                                found = True
                            except (TypeError, ValueError):
                                pass
                if found:
                    return total
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return None


def _sum_json_field(data, names: tuple[str, ...]):
    """Sum every numeric field named ``names`` (with an optional {value, unit}
    wrapper) anywhere in an amd-smi JSON document. amd-smi's exact nesting moves
    between ROCm releases; the field names are the stable part."""
    total = 0
    found = False

    def walk(node):
        nonlocal total, found
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in names:
                    number = value.get("value") if isinstance(value, dict) else value
                    try:
                        total += int(float(number))
                        found = True
                        continue
                    except (TypeError, ValueError):
                        pass
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    return total if found else None


# --------------------------------------------------------------------------- #
# rocprofv3                                                                    #
# --------------------------------------------------------------------------- #

_HELP_CACHE: str | None = None


def _rocprofv3_help() -> str:
    global _HELP_CACHE
    if _HELP_CACHE is None:
        try:
            _HELP_CACHE = subprocess.run(
                ["rocprofv3", "--help"], capture_output=True, text=True, timeout=60
            ).stdout
        except (OSError, subprocess.SubprocessError):
            _HELP_CACHE = ""
    return _HELP_CACHE


def rocprofv3_prefix(out_dir: Path):
    """The rocprofv3 wrapper for a serving launch, or [] when unavailable.

    Flags are probed against this rocprofv3's --help rather than assumed: the
    vLLM ROCm image pins whatever sdk it pins, and one unsupported flag would
    kill the server at launch — a tracing option must never be able to do that.

    Requested planes: kernel dispatch timeline, HIP runtime calls (this is
    where event waits/barriers appear at the API level), memory copies and
    allocations, RCCL ops. csv output for programmatic taxonomy, pftrace for
    timeline viewing. Explicitly NOT --pmc: counters serialize kernels.
    """
    if os.environ.get(ENV_NO_ROCPROF, "").strip() not in ("", "0"):
        console.log(f"[yellow]{ENV_NO_ROCPROF} set; launching without rocprofv3[/yellow]")
        return []
    if shutil.which("rocprofv3") is None:
        console.log("[yellow]rocprofv3 not found; kernel plane will be missing for this run[/yellow]")
        return []

    help_text = _rocprofv3_help()
    wanted = [
        "--kernel-trace",
        "--hip-trace",
        "--memory-copy-trace",
        "--memory-allocation-trace",
        "--rccl-trace",
        "--marker-trace",
    ]
    flags = [flag for flag in wanted if flag in help_text]
    skipped = [flag for flag in wanted if flag not in help_text]
    if skipped:
        console.log(f"[yellow]rocprofv3 here lacks {' '.join(skipped)}; continuing without[/yellow]")
    if not flags:
        console.log("[yellow]rocprofv3 supports none of the requested trace flags; skipping wrap[/yellow]")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    formats = ["csv"] + (["pftrace"] if "pftrace" in help_text else [])
    return (
        ["rocprofv3", *flags, "--output-format", *formats, "-d", str(out_dir), "-o", "trace", "--"]
    )


# --------------------------------------------------------------------------- #
# amd-smi sampler                                                              #
# --------------------------------------------------------------------------- #


class AmdSmiSampler:
    """Background GPU telemetry at SAMPLE_INTERVAL_S, JSONL into the run dir.

    Fast path: the ``amdsmi`` Python library (ships with ROCm, preinstalled in
    the vLLM ROCm image) — direct calls easily hold 100 ms. Fallback: an
    ``amd-smi metric --json`` subprocess loop, which is slower per tick; each
    record carries its own timestamp so the achieved cadence is in the data,
    not assumed.

    Per tick: GPU busy %, VRAM used, power draw + cap, energy accumulator,
    GFX/MEM clocks, junction + memory temps. Every SLOW_SAMPLE_EVERY ticks the
    full gpu-metrics table (throttle/violation status, PCIe bandwidth and
    replays, xGMI per-link accumulators) is snapshotted whole.

    Every library call is individually tolerant: amdsmi renames fields between
    ROCm releases, and one missing getter must cost that column, not the run.
    """

    def __init__(self, out_path: Path, interval_s: float = SAMPLE_INTERVAL_S):
        self.out_path = Path(out_path)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="amd-smi-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    # -- internals ---------------------------------------------------------- #

    def _run(self):
        try:
            import amdsmi  # type: ignore

            self._run_lib(amdsmi)
            return
        except Exception as exc:  # noqa: BLE001 - any lib failure -> CLI fallback
            console.log(f"[yellow]amdsmi library unavailable ({exc}); falling back to amd-smi CLI[/yellow]")
        self._run_cli()

    @staticmethod
    def _call(fn, *args):
        try:
            return fn(*args)
        except Exception:  # noqa: BLE001 - per-field tolerance is the contract
            return None

    def _run_lib(self, amdsmi):
        amdsmi.amdsmi_init()
        try:
            handles = amdsmi.amdsmi_get_processor_handles()
            caps = [
                self._call(amdsmi.amdsmi_get_power_cap_info, handle) for handle in handles
            ]
            tick = 0
            with self.out_path.open("a", encoding="utf-8") as out:
                out.write(json.dumps({"kind": "meta", "source": "amdsmi-lib",
                                      "interval_s": self.interval_s,
                                      "power_caps": _plain(caps)}) + "\n")
                while not self._stop.is_set():
                    started = time.monotonic()
                    sample = {"kind": "sample", "t_ns": time.time_ns(), "gpus": []}
                    for index, handle in enumerate(handles):
                        gpu = {"gpu": index}
                        activity = self._call(amdsmi.amdsmi_get_gpu_activity, handle)
                        vram = self._call(amdsmi.amdsmi_get_gpu_vram_usage, handle)
                        power = self._call(amdsmi.amdsmi_get_power_info, handle)
                        energy = self._call(amdsmi.amdsmi_get_energy_count, handle)
                        gpu.update({
                            "activity": _plain(activity),
                            "vram": _plain(vram),
                            "power": _plain(power),
                            "energy": _plain(energy),
                        })
                        clock_type = getattr(amdsmi, "AmdSmiClkType", None)
                        if clock_type is not None:
                            gpu["clk_gfx"] = _plain(self._call(
                                amdsmi.amdsmi_get_clock_info, handle, clock_type.GFX))
                            gpu["clk_mem"] = _plain(self._call(
                                amdsmi.amdsmi_get_clock_info, handle, clock_type.MEM))
                        temp_type = getattr(amdsmi, "AmdSmiTemperatureType", None)
                        temp_metric = getattr(amdsmi, "AmdSmiTemperatureMetric", None)
                        if temp_type is not None and temp_metric is not None:
                            gpu["temp_junction"] = self._call(
                                amdsmi.amdsmi_get_temp_metric, handle,
                                getattr(temp_type, "JUNCTION", getattr(temp_type, "HOTSPOT", None)),
                                temp_metric.CURRENT)
                            gpu["temp_vram"] = self._call(
                                amdsmi.amdsmi_get_temp_metric, handle,
                                getattr(temp_type, "VRAM", None), temp_metric.CURRENT)
                        if tick % SLOW_SAMPLE_EVERY == 0:
                            gpu["gpu_metrics"] = _plain(self._call(
                                amdsmi.amdsmi_get_gpu_metrics_info, handle))
                        sample["gpus"].append(gpu)
                    out.write(json.dumps(sample) + "\n")
                    out.flush()
                    tick += 1
                    self._stop.wait(max(0.0, self.interval_s - (time.monotonic() - started)))
        finally:
            self._call(amdsmi.amdsmi_shut_down)

    def _run_cli(self):
        binary = "amd-smi" if shutil.which("amd-smi") else (
            "rocm-smi" if shutil.which("rocm-smi") else None)
        if binary is None:
            console.log("[red]No amd-smi or rocm-smi on PATH; GPU telemetry disabled[/red]")
            return
        argv = ([binary, "metric", "--json"] if binary == "amd-smi"
                else [binary, "--showuse", "--showmemuse", "--showpower", "--showtemp", "--json"])
        xgmi_argv = [binary, "metric", "--xgmi", "--json"] if binary == "amd-smi" else None
        tick = 0
        with self.out_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"kind": "meta", "source": f"{binary}-cli",
                                  "interval_s": self.interval_s,
                                  "note": "CLI fallback; cadence is whatever each invocation costs"}) + "\n")
            while not self._stop.is_set():
                started = time.monotonic()
                record = {"kind": "sample", "t_ns": time.time_ns()}
                record["metric"] = _cli_json(argv)
                if xgmi_argv and tick % SLOW_SAMPLE_EVERY == 0:
                    record["xgmi"] = _cli_json(xgmi_argv)
                record["sample_cost_s"] = round(time.monotonic() - started, 4)
                out.write(json.dumps(record) + "\n")
                out.flush()
                tick += 1
                self._stop.wait(max(0.0, self.interval_s - (time.monotonic() - started)))


def _cli_json(argv):
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return json.loads(proc.stdout) if proc.returncode == 0 else {"error": proc.stderr[-500:]}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"error": str(exc)}


def _plain(value):
    """amdsmi returns dicts/lists of ctypes-ish values; force JSON-serializable."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return repr(value)


def nic_counters():
    """ethtool -S + RDMA hw_counters snapshot, for before/after each experiment.

    Multi-node fabric is not assumed InfiniBand — this records whatever the
    NICs actually are. Cheap enough to run unconditionally; TP within one node
    still exercises no NIC, and the delta will simply be noise-level.
    """
    snapshot = {"t_ns": time.time_ns(), "ethtool": {}, "rdma_hw_counters": {}}
    net = Path("/sys/class/net")
    if net.is_dir() and shutil.which("ethtool"):
        for iface in sorted(net.iterdir()):
            name = iface.name
            if name == "lo":
                continue
            try:
                proc = subprocess.run(["ethtool", "-S", name],
                                      capture_output=True, text=True, timeout=15)
                if proc.returncode == 0:
                    snapshot["ethtool"][name] = proc.stdout
            except (OSError, subprocess.SubprocessError):
                continue
    infiniband = Path("/sys/class/infiniband")
    if infiniband.is_dir():
        for device in sorted(infiniband.iterdir()):
            for counters in device.glob("ports/*/hw_counters"):
                values = {}
                for counter in counters.iterdir():
                    try:
                        values[counter.name] = counter.read_text().strip()
                    except OSError:
                        continue
                snapshot["rdma_hw_counters"][f"{device.name}:{counters.parent.name}"] = values
    return snapshot
