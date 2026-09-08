"""Command line entry point for the `rex` runtime experiment harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import shutil
import sys
from pathlib import Path

from rich.console import Console

from . import __version__
from .analysis import (
    analyze_capture,
    discover_capture_dirs,
    overhead_rows,
    print_capture_analysis,
    render_overhead,
)
from .banner import harness_banner
from .checks import DEFAULT_EXPERIMENTS_DIR, DEFAULT_HF_HOME, DEFAULT_REPORTS_DIR, run_preflight
from .experiments import (
    BUNDLED_CONFIG,
    DEFAULT_CONFIG_NAME,
    TEMPLATES,
    run_all_experiments,
    template_path,
)
from .install import (
    DEFAULT_CLONE_DIR,
    RUNTIME_REPO_URL,
    default_extras,
    install_runtime,
    parse_extras,
)

console = Console()


def cmd_check(args) -> int:
    # Printed directly, not via rich: the banner already carries its own ANSI
    # colour, and rich would re-parse those escapes as literal text.
    print(harness_banner(__version__))
    run_preflight(
        install=not args.no_install,
        hf_home=args.hf_home,
        experiments_dir=args.experiments_dir,
        reports_dir=args.reports_dir,
    )
    console.print("[bold green]Host ready.[/bold green] Run [bold]rex run[/bold] to start experiments.")
    return 0


def cmd_install(args) -> int:
    repo, extras = install_runtime(
        dest=args.dest,
        url=args.url,
        ref=args.ref,
        extras=parse_extras(args.extras),
        editable=not args.no_editable,
        constraints=not args.no_constraints,
        update=not args.no_update,
        gitm_install=False if args.no_gitm_install else (True if args.gitm_install else None),
        gitm_install_args=shlex.split(args.gitm_install_args or ""),
    )
    console.print(
        f"[bold green]Runtime installed from {repo}.[/bold green] "
        "Run [bold]rex check[/bold] to provision the host."
    )
    return 0


def cmd_run(args) -> int:
    results = asyncio.run(
        run_all_experiments(
            args.config,
            install=not args.no_install,
            output_dir=args.output_dir,
            hf_home=args.hf_home,
            experiments_dir=args.experiments_dir,
            reports_dir=args.reports_dir,
        )
    )
    # A sweep keeps going past a failed experiment (the rest of the matrix is
    # still worth having), but the process must not report success for it —
    # a k8s Job that shows Succeeded on a dead server is invisible breakage.
    # Skipped rows (e.g. NVIDIA-only capture on ROCm) are not failures.
    failed = [
        r["name"] for r in results
        if not r.get("success") and r.get("status") != "skipped"
    ]
    if failed:
        console.print(
            f"[bold red]{len(failed)} experiment step(s) failed:[/bold red] "
            + ", ".join(dict.fromkeys(failed))
        )
        return 1
    return 0


def cmd_analyze(args) -> int:
    """Re-read capture directories that already exist -- no GPU, no server."""
    capture_dirs = discover_capture_dirs(args.paths)
    if not capture_dirs:
        console.print(
            f"[yellow]No capture directories found under {', '.join(str(p) for p in args.paths)}. "
            f"A capture directory holds run_manifest.json / kernel_breakdown.json.[/yellow]"
        )
        return 1

    records = []
    for capture_dir in capture_dirs:
        console.rule(str(capture_dir))
        record = analyze_capture(
            capture_dir,
            layerwise=not args.no_layerwise,
            phase=not args.no_phase,
        )
        print_capture_analysis(record)
        records.append(record)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2), encoding="utf-8")
        console.print(f"[green]Wrote analysis for {len(records)} capture(s) to {args.json}[/green]")
    return 0


def cmd_compare(args) -> int:
    """Tracing-overhead table across capture directories.

    Arms are grouped on what each run reported in serving_summary.json, not on the
    directory name -- a directory called `ovh-off-1` that ran with the collector
    attached is exactly the failure this table exists to expose.
    """
    capture_dirs = discover_capture_dirs(args.paths)
    if not capture_dirs:
        console.print("[yellow]No capture directories found.[/yellow]")
        return 1

    records = [
        analyze_capture(capture_dir, layerwise=False, phase=False)
        for capture_dir in capture_dirs
    ]
    rows = overhead_rows(records)
    console.print(render_overhead(rows))

    missing = sum(row["n_missing_throughput"] for row in rows)
    if missing:
        console.print(
            f"[yellow]{missing} run(s) reported no server-side throughput. vLLM's "
            f"/metrics is the only source of tokens/s here -- the client-side summary "
            f"carries latency percentiles but no token counts.[/yellow]"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        console.print(f"[green]Wrote comparison to {args.json}[/green]")
    return 0


def cmd_init(args) -> int:
    source = template_path(args.template)
    destination = args.path
    if destination.exists() and not args.force:
        console.print(
            f"[yellow]{destination} already exists. Pass --force to overwrite.[/yellow]"
        )
        return 1
    shutil.copyfile(source, destination)
    console.print(f"[green]Wrote starter manifest to {destination}[/green]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rex",
        description="Runtime experiment harness for vLLM / guidellm on NVLink GPU clusters.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_setup_flags(sub):
        sub.add_argument(
            "--hf-home",
            type=Path,
            default=None,
            help=f"Hugging Face cache directory (default: {DEFAULT_HF_HOME}).",
        )
        sub.add_argument(
            "--experiments-dir",
            type=Path,
            default=None,
            help=f"Directory for per-experiment results (default: ./{DEFAULT_EXPERIMENTS_DIR}).",
        )
        sub.add_argument(
            "--reports-dir",
            type=Path,
            default=None,
            help=f"Directory for guidellm JSON reports (default: {DEFAULT_REPORTS_DIR}).",
        )
        sub.add_argument(
            "--no-install",
            action="store_true",
            help="Skip installing torch/vllm/guidellm/huggingface-hub; verify and set up only.",
        )

    check = subparsers.add_parser(
        "check",
        help="Verify CUDA and topology, install runtime deps, and prepare the workspace.",
    )
    add_setup_flags(check)
    check.set_defaults(func=cmd_check)

    run = subparsers.add_parser("run", help="Run the experiment suite from a YAML manifest.")
    run.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_NAME),
        help=f"Path to the experiments manifest (default: ./{DEFAULT_CONFIG_NAME}).",
    )
    run.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots and experiment_results.json (default: current directory).",
    )
    add_setup_flags(run)
    run.set_defaults(func=cmd_run)

    install = subparsers.add_parser(
        "install",
        help="Clone the Git.M runtime repository and install its dependencies.",
    )
    install.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_CLONE_DIR,
        help=f"Where to clone the repository (default: ./{DEFAULT_CLONE_DIR}).",
    )
    install.add_argument(
        "--url",
        default=RUNTIME_REPO_URL,
        help=f"Repository to clone (default: {RUNTIME_REPO_URL}).",
    )
    install.add_argument(
        "--ref",
        default=None,
        help="Branch, tag, or commit to check out (default: the repository default branch).",
    )
    install.add_argument(
        "--extras",
        default=None,
        help=(
            "Comma-separated extras to install, or '' for none "
            f"(default on this host: {','.join(default_extras())})."
        ),
    )
    install.add_argument(
        "--no-editable",
        action="store_true",
        help="Install a regular copy instead of an editable checkout.",
    )
    install.add_argument(
        "--no-constraints",
        action="store_true",
        help="Ignore the repository's constraints.txt pins.",
    )
    install.add_argument(
        "--gitm-install",
        action="store_true",
        help="Run `gitm install` even where nvidia-smi is absent.",
    )
    install.add_argument(
        "--no-gitm-install",
        action="store_true",
        help="Skip `gitm install` (driver-matched CUPTI, pinned vLLM/torch, tracer shim).",
    )
    install.add_argument(
        "--gitm-install-args",
        default=None,
        help="Extra flags for `gitm install`, e.g. \"--skip-apt --skip-stack\".",
    )
    install.add_argument(
        "--no-update",
        action="store_true",
        help="Reuse an existing checkout as-is instead of fetching the latest commit.",
    )
    install.set_defaults(func=cmd_install)

    analyze = subparsers.add_parser(
        "analyze",
        help="Read capture directories: kernel buckets, NVTX layer/op tables, prefill vs decode.",
    )
    analyze.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Capture directories, or a directory holding them (e.g. experiments/<id>).",
    )
    analyze.add_argument(
        "--no-layerwise",
        action="store_true",
        help="Skip the NVTX layer/op tables (a pass over the trace).",
    )
    analyze.add_argument(
        "--no-phase",
        action="store_true",
        help="Skip the prefill/decode split (needs gitm importable).",
    )
    analyze.add_argument("--json", type=Path, default=None, help="Write the analysis here.")
    analyze.set_defaults(func=cmd_analyze)

    compare = subparsers.add_parser(
        "compare",
        help="Compare capture arms: throughput and TPOT per tracing state.",
    )
    compare.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Capture directories, or a directory holding them.",
    )
    compare.add_argument("--json", type=Path, default=None, help="Write the comparison here.")
    compare.set_defaults(func=cmd_compare)

    init = subparsers.add_parser("init", help="Write a starter experiments.yaml.")
    init.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path(DEFAULT_CONFIG_NAME),
        help=f"Where to write the manifest (default: ./{DEFAULT_CONFIG_NAME}).",
    )
    init.add_argument(
        "--template",
        default="serving",
        choices=sorted(TEMPLATES),
        help=(
            "Which starter manifest to write. `serving` and `cudagraph-spec` drive "
            "guidellm; `capture`, `overhead` and `cudagraph-spec-capture` drive "
            "`gitm capture serve` (default: serving)."
        ),
    )
    init.add_argument("--force", action="store_true", help="Overwrite an existing file.")
    init.set_defaults(func=cmd_init)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        console.print(f"[bold red]Experiment harness failed:[/bold red] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
