# Experiment Harness

A Python-based experiment harness for NVIDIA H100 clusters with NVLink.

## Features

- Detects GPU topology and NVLink connectivity
- Installs `vllm`, `guidellm`, and plotting dependencies
- Loads experiments from a YAML manifest
- Launches VLLM with EP/DP/TP/PP settings
- Runs `guidellm` workloads sequentially
- Plots results with Plotly asynchronously

## Quick Start

```bash
python3 -m pip install -r requirements.txt
python3 run_experiments.py --config experiments.yaml
```

## Config

The sample `experiments.yaml` contains experiment definitions with model, prompt, and parallelism settings.
