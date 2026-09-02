# Runtime Experiment Harness

A Python experiment harness for benchmarking vLLM with guidellm on NVIDIA
H100 / NVLink clusters. Ships as the `rex` command.

## Install

```bash
pip install runtime-experiment-harness            # harness only
pip install "runtime-experiment-harness[plot]"    # + Plotly HTML result plots
pip install "runtime-experiment-harness[gpu]"     # + vllm and guidellm (CUDA hosts only)
```

`vllm` and `guidellm` are intentionally optional. They are large, CUDA-specific,
and will not build on a machine without an NVIDIA toolchain — making them
required would break `pip install` on a laptop. The harness invokes both as
subprocesses, so install the `gpu` extra on the cluster (or let `rex run
--install` fetch them at runtime).

## Usage

```bash
rex install                     # clone the Git.M runtime and install its deps
rex init                        # write a starter experiments.yaml
rex check                       # verify the host and provision it
rex run                         # provision, then run every experiment
rex analyze <capture-dir>       # read a capture back: kernels, layers, phases
rex compare <capture-dirs>      # throughput per tracing arm
```

`rex install` clones <https://github.com/GitM-Labs/runtime.git> into `./runtime`
and pip-installs it into the interpreter running the harness, so the `gitm` CLI
is importable from the same environment. It uses the repository's
`constraints.txt` pins, and installs an editable checkout so a `git pull` in
`runtime/` is picked up without reinstalling.

Which extras it selects depends on the host: a box with `nvidia-smi` on PATH
gets `gpu,vllm,bench` (RAPIDS, CUPTI/NVML, and the vLLM decode workload, with
NVIDIA's package index added since cuDF and CuPy are not on PyPI); anywhere else
gets `bench`, because the GPU wheels are Linux/CUDA-only and would fail to
resolve. Override with `--extras`.

```bash
rex install --dest /workspace/runtime     # clone somewhere other than ./runtime
rex install --ref main                    # pin a branch, tag, or commit
rex install --extras gpu,vllm,dev         # choose extras explicitly
rex install --extras ""                   # base dependencies only
rex install --no-constraints              # ignore the repo's version pins
rex install --no-editable                 # install a copy, not a live checkout
rex install --no-update                   # reuse an existing checkout as-is
```

On a CUDA host it then runs `gitm install` (driver-matched CUPTI, pinned
vLLM/torch, and the tracer shim every capture depends on — without it `gitm
capture serve` fails preflight at the injection-lib check). Off a CUDA host that
step is skipped; `--no-gitm-install` skips it anywhere, and `--gitm-install-args`
forwards flags such as `--skip-apt`.

Re-running `rex install` fetches the latest commit into an existing checkout
rather than re-cloning. It never deletes the destination: a directory holding
something else — or a checkout of a different repository — is reported as an
error so you can pick a different `--dest`.

Set `HF_TOKEN` before running anything against a gated checkpoint. The harness
warns when it is missing, because the failure otherwise lands minutes into the
run, during weight download, well after preflight has passed:

```bash
export HF_TOKEN=hf_...
```

## Kernel captures

An experiment with a `capture:` block is driven by `gitm capture serve` instead
of guidellm. The two are exclusive by construction: the CUDA driver reads
`CUDA_INJECTION64_PATH` exactly once, at CUDA init, so a server this harness
started itself can never be traced afterwards. gitm owns the server for a
capture; the harness owns the sweep around it.

```bash
rex init experiments-capture.yaml --template capture
rex run --config experiments-capture.yaml
```

```yaml
experiments:
  - name: qwen-eager
    model: Qwen/Qwen3.6-35B-A3B
    vllm_args: ["--trust-remote-code", "--max-model-len", "16384", "--enforce-eager"]
    capture:
      requests: 64
      concurrency: 8
      input_tokens: 512
      output_tokens: 512
      warmup: 8
      arms: ["cupti"]      # off | cupti | nvtx
      repeat: 1
```

Each `(arm, repetition)` gets its own directory under
`experiments/<experiment-id>/`, holding what gitm writes (`trace.jsonl`,
`kernel_breakdown.json`, `serving_summary.json`, `server.log`, `preflight.json`)
plus a `harness_record.json` saying which arm and which sweep it belongs to.

### The three arms

|arm    |collector                                    |answers                          |
|-------|---------------------------------------------|---------------------------------|
|`off`  |none — the injection variables are *cleared*  |the untraced baseline            |
|`cupti`|kernels, memcpy, sync                         |what ran, by taxonomy bucket     |
|`nvtx` |`cupti` + the correlation chain and vLLM ranges|which layer and which op it was |

Only the `nvtx` arm can attribute a kernel to a layer: every other arm's kernels
come back with `range_op: null`, because an anonymous `nvjet` GEMM is not
identifiable from its name. Never put `--enable-layerwise-nvtx-tracing` in
`vllm_args` — gitm's `--nvtx` adds it to exactly the arm that collects the
ranges. Setting it by hand pushes ranges in every arm, including the ones whose
throughput is supposed to be range-free, and the cost lands in the comparison
without appearing anywhere in the configuration. The same goes for
`GITM_TRACE_NVTX` and `NVTX_INJECTION64_PATH`: `--nvtx` sets both.

### What `rex run` prints per capture

* **kernel buckets** — time by taxonomy bucket, from `kernel_breakdown.json`.
* **layer / op tables** — `nvtx` arms only, streamed from the merged trace.
  Ops outside the decoder stack (`logits_processor`, `embed_tokens`) have no
  layer, so they appear in the op table and not the layer table.
* **prefill vs decode** — gitm's phase classifier, reported with its attribution
  stats. Trust it only as far as those go: most kernels are byte-identical in
  both phases and inherit the phase of the nearest kernel that names one. A
  median gap of microseconds means the neighbour is in the same engine step;
  milliseconds means the inference crossed a step boundary, and under chunked
  prefill a single step genuinely mixes both phases.

`rex analyze <dir>` reprints all of it from an existing capture directory — no
GPU, no server, and it accepts either the capture directories themselves or a
parent holding them.

## Tracing overhead

```bash
rex init experiments-overhead.yaml --template overhead
rex run --config experiments-overhead.yaml
rex compare experiments/tracer-overhead-gpu1
```

Three arms, one server command, one client workload, one seed — the only
difference is the state of the collector. Repetitions interleave the arms
(`off, cupti, nvtx, off, ...`) rather than running three of each in a row, so
anything that drifts over the session doesn't land entirely on one arm.

The table is grouped on the tracing state each run *reported* in
`serving_summary.json`, not on the arm the harness asked for. Those disagree
exactly when the measurement is worthless — an injection variable inherited from
the shell attaches the collector to the arm whose purpose is not to have one —
and `rex run` flags the mismatch per run rather than averaging it in.

Throughput comes from vLLM's own `/metrics`, differenced across the window.
There is no client-side tokens/s to read: the client block reports latency
percentiles and goodput but carries no token counts, so a run whose scrape
failed reports no throughput rather than a zero that would read as catastrophic
overhead.

## CUDA graphs and speculative decoding

Two manifests over the same eight arms — four `cudagraph_mode` settings, three
MTP draft depths, and eager as the floor:

```bash
rex init sweep.yaml --template cudagraph-spec          # guidellm: how fast it serves
rex init sweep.yaml --template cudagraph-spec-capture  # gitm: what it actually ran
```

Check the resolved mode before comparing anything. vLLM *resolves*
`cudagraph_mode` and may downgrade it — a hybrid model with linear-attention
layers constrains `PIECEWISE` — so an arm can ask for one mode and run another
without failing:

```bash
grep -h "cudagraph_mode" experiments/*/*/server.log
```

Two arms that silently resolved to the same mode differ only by noise, and that
reads as "the mode does not matter". Under graphs, expect `gpu_active_share` to
fall without the GPU doing less: replayed graph work is not a launch, so it is
not attributed. That is why the graph arms are read against serving throughput
and the eager arm carries the layer attribution.

`rex check` prepares the box in five steps and stops at the first failure:

1. Verify CUDA >= 13.0 (vLLM dropped everything older)
2. Print GPU count and the `nvidia-smi` NVLink topology matrix
3. Install `torch`, `vllm`, `guidellm`, `huggingface-hub`, `plotly`
4. Create the HF cache, `experiments/`, and `/workspace/guidellm_reports`
5. Point `HF_HOME` at the cache

`rex run` performs the same five steps, then runs the sweep. Useful flags on
both:

```bash
rex check --no-install                    # verify and set up, skip pip
rex check --hf-home /mnt/models           # default is /workspace/hf_hub
rex run --config sweep.yaml --output-dir ./results
rex run --experiments-dir /workspace/experiments
```

`python -m runtime_harness` works identically to `rex` if the console script is
not on your PATH.

## Results

Each guidellm run is written per-experiment as:

```text
experiments/<experiment-id>/<timestamp>_<experiment-id>.json
```

where `<experiment-id>` is `<name>-gpu<N>` — so one manifest entry swept across
2, 4, and 8 GPUs produces three directories. Timestamps are UTC and
filename-safe (`20260803T142305Z`); colons are avoided because they are illegal
in filenames on macOS and Windows and need quoting in every shell.

guidellm's own JSON report is written to `/workspace/guidellm_reports` as
`<timestamp>_<experiment-id>_guidellm.json`, and a copy is placed beside the
harness record so each run directory is self-contained. A manifest that sets
`--output kind=json,path=...` explicitly on its `guidellm_command` keeps that
path — an explicit config value beats the default.

## vLLM startup

`vllm_args` from the manifest are passed through verbatim, with
`--tensor-parallel-size` and `--enable-expert-parallel` filled in from the
`tp`/`ep` fields when not already present. The harness then polls
`http://localhost:<port>/health` until the server actually accepts traffic
rather than sleeping a fixed interval — a cold HF cache can take many minutes to
load a large checkpoint. `--port` is read from `vllm_args` (default 8000). If
the server dies or fails to come up within 15 minutes, that experiment is
recorded as failed and the sweep moves on.

A combined `experiment_results.json` and a `plots/` folder also land in the
working directory (override with `--output-dir`).

### `HF_HOME` and your shell

`rex` exports `HF_HOME` into its own process, so every vLLM and guidellm
subprocess it launches inherits the cache location. A child process cannot
change its parent's environment, so this does **not** persist into your shell —
`rex` prints the matching `export` line if you want it there too.

## Requirements

- Python 3.10+
- CUDA 13.0 or newer — vLLM has dropped support for anything older, so `rex
  check` and `rex run` both refuse to proceed below that and tell you what was
  detected.
- NVIDIA drivers with `nvidia-smi` on `PATH`

## Config

`experiments.yaml` defines the sweep: `gpu_counts` to iterate over, then one
entry per experiment with model, prompt, and EP/DP/TP/PP parallelism settings.
`vllm_args` and `guidellm_args` are passed through verbatim, so any flag the
underlying tools accept works without changes here. `rex init` writes a
worked example to start from.

## Development

```bash
pip install -e ".[dev,plot]"
pytest
```

## Release

```bash
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```
