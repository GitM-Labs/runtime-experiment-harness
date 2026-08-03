"""Command line entry point for the `rex` runtime experiment harness."""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

from rich.console import Console

from . import __version__
from .banner import harness_banner
from .checks import DEFAULT_EXPERIMENTS_DIR, DEFAULT_HF_HOME, DEFAULT_REPORTS_DIR, run_preflight
from .experiments import BUNDLED_CONFIG, DEFAULT_CONFIG_NAME, run_all_experiments

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


def cmd_run(args) -> int:
    asyncio.run(
        run_all_experiments(
            args.config,
            install=not args.no_install,
            output_dir=args.output_dir,
            hf_home=args.hf_home,
            experiments_dir=args.experiments_dir,
            reports_dir=args.reports_dir,
        )
    )
    return 0


def cmd_init(args) -> int:
    destination = args.path
    if destination.exists() and not args.force:
        console.print(
            f"[yellow]{destination} already exists. Pass --force to overwrite.[/yellow]"
        )
        return 1
    shutil.copyfile(BUNDLED_CONFIG, destination)
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

    init = subparsers.add_parser("init", help="Write a starter experiments.yaml.")
    init.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path(DEFAULT_CONFIG_NAME),
        help=f"Where to write the manifest (default: ./{DEFAULT_CONFIG_NAME}).",
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
