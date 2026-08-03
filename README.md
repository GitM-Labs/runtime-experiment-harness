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
rex init                        # write a starter experiments.yaml
rex check                       # verify the host and provision it
rex run                         # provision, then run every experiment
```

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
