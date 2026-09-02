"""Clone the Git.M runtime repository and install its dependencies.

`rex install` is the "get this box from bare to buildable" step that runs before
`rex check`: it puts the runtime source tree on disk and pip-installs it, so the
`gitm` CLI the experiments drive is importable from the same interpreter as the
harness.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from rich.console import Console

from .checks import run_command

console = Console()

RUNTIME_REPO_URL = "https://github.com/GitM-Labs/runtime.git"
DEFAULT_CLONE_DIR = Path("runtime")

# cuDF and CuPy are published on NVIDIA's own index, not PyPI, so the `gpu`
# extra cannot resolve without it. Harmless when that extra is not selected;
# pip only reaches for the extra index on names it cannot find on PyPI.
NVIDIA_INDEX_URL = "https://pypi.nvidia.com"
GPU_EXTRA = "gpu"

# The runtime's own split (see its pyproject): `gpu` pulls RAPIDS + CUPTI +
# NVML, `vllm` pulls the decode workload, `bench` is the CPU-only Parquet/pandas
# path that installs anywhere.
GPU_HOST_EXTRAS = ("gpu", "vllm", "bench")
CPU_HOST_EXTRAS = ("bench",)

# Pinned versions the runtime's test suite is green against. Its Dockerfile
# installs with this file, so honouring it here gets the same stack on the host.
CONSTRAINTS_NAME = "constraints.txt"


def default_extras(gpu_host=None):
    """Extras to install when the caller names none.

    A CUDA box gets the full GPU stack; anywhere else that stack cannot resolve
    (Linux-only wheels, an NVIDIA index that has nothing for macOS), so a laptop
    gets the CPU path instead of a failed install.
    """
    if gpu_host is None:
        gpu_host = shutil.which("nvidia-smi") is not None
    return list(GPU_HOST_EXTRAS if gpu_host else CPU_HOST_EXTRAS)


def parse_extras(raw):
    """Split a comma/space separated --extras value into a deduplicated list."""
    if raw is None:
        return None
    parts = [part.strip() for chunk in raw.split(",") for part in chunk.split()]
    seen = {}
    for part in parts:
        if part:
            seen[part] = None
    return list(seen)


def remote_url(repo: Path):
    """Return the origin URL of an existing checkout, or None if it has none."""
    try:
        output = run_command(
            ["git", "-C", repo, "remote", "get-url", "origin"],
            capture_output=True,
        )
    except Exception:
        return None
    return (output or "").strip() or None


def same_repo(left: str, right: str) -> bool:
    """Compare remotes ignoring the cosmetic differences between clone URLs."""

    def normalise(url: str) -> str:
        url = url.strip().rstrip("/")
        if url.endswith(".git"):
            url = url[: -len(".git")]
        url = url.replace("git@github.com:", "https://github.com/")
        return url.lower()

    return normalise(left) == normalise(right)


def clone_or_update(dest=None, url=RUNTIME_REPO_URL, ref=None, update=True) -> Path:
    """Clone the runtime repo to `dest`, or update the checkout already there.

    Re-running install is the normal way to pick up a new commit, so an existing
    checkout of the same repo is updated rather than treated as an error. A
    directory holding something else is left strictly alone -- this function
    never deletes a path it did not create.
    """
    if shutil.which("git") is None:
        raise RuntimeError("git not found in PATH. Install git before running `rex install`.")

    dest = Path(dest) if dest is not None else DEFAULT_CLONE_DIR

    if (dest / ".git").is_dir():
        existing = remote_url(dest)
        if existing is not None and not same_repo(existing, url):
            raise RuntimeError(
                f"{dest} is a checkout of {existing}, not {url}. "
                "Pass --dest to clone somewhere else."
            )
        if not update:
            console.print(f"[dim]Reusing existing checkout at {dest}[/dim]")
            return dest
        console.rule(f"Updating {dest}")
        # Fetch by ref name so a --ref that is a tag or a bare commit works the
        # same as a branch, then detach onto FETCH_HEAD rather than merging --
        # the checkout is a build input, not somewhere anyone commits.
        run_command(["git", "-C", dest, "fetch", "--tags", "origin", ref or "HEAD"])
        run_command(["git", "-C", dest, "checkout", "--force", ref or "FETCH_HEAD"])
        return dest

    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError(
            f"{dest} already exists and is not a git checkout. "
            "Remove it or pass --dest to clone somewhere else."
        )

    console.rule(f"Cloning {url}")
    # A commit SHA is not a valid --branch argument, so a pinned SHA is cloned
    # first and checked out afterwards; a branch or tag is selected up front.
    pinned_sha = ref is not None and looks_like_sha(ref)
    command = ["git", "clone"]
    if ref is not None and not pinned_sha:
        command += ["--branch", ref]
    command += [url, dest]
    run_command(command)
    if pinned_sha:
        run_command(["git", "-C", dest, "checkout", "--force", ref])
    return dest


def looks_like_sha(ref: str) -> bool:
    return len(ref) >= 7 and all(char in "0123456789abcdef" for char in ref.lower())


def build_pip_command(repo: Path, extras=None, editable=True, constraints=True):
    """Assemble the pip invocation that installs the cloned runtime."""
    extras = list(extras or [])
    # Resolved, so pip reads "/abs/path[gpu]" as a local directory plus extras
    # rather than as the name of a project on PyPI called "runtime".
    target = str(Path(repo).resolve())
    if extras:
        target = f"{target}[{','.join(extras)}]"

    command = [sys.executable, "-m", "pip", "install"]
    if editable:
        command.append("-e")
    command.append(target)

    if constraints:
        constraints_file = Path(repo).resolve() / CONSTRAINTS_NAME
        if constraints_file.is_file():
            command += ["-c", str(constraints_file)]

    if GPU_EXTRA in extras:
        command += ["--extra-index-url", NVIDIA_INDEX_URL]

    return command


def install_dependencies(repo: Path, extras=None, editable=True, constraints=True):
    if not (repo / "pyproject.toml").is_file():
        raise RuntimeError(f"{repo} has no pyproject.toml -- the clone looks incomplete.")

    console.rule("Installing runtime dependencies")
    command = build_pip_command(repo, extras=extras, editable=editable, constraints=constraints)
    run_command(command)
    console.log("[green]Runtime installed successfully[/green]")
    return command


def gitm_command(subcommand):
    """`gitm <subcommand>`, via the console script when it is on PATH.

    The fallback runs the module through *this* interpreter, which is the one the
    package was just installed into. A `gitm` on PATH from a different environment
    would prepare that environment's CUDA stack instead of this one's, and nothing
    about the output would say so.
    """
    if shutil.which("gitm") is not None:
        return ["gitm", *subcommand]
    return [sys.executable, "-m", "gitm.cli", *subcommand]


def run_gitm_install(extra_args=None, gpu_host=None):
    """Run `gitm install`: driver-matched CUPTI, pinned vLLM/torch, tracer shim.

    Skipped off a CUDA host, where there is no driver to match and nothing it does
    would apply. This is the step that builds the injection library every capture
    depends on -- without it `gitm capture serve` fails preflight at the
    injection-lib check.
    """
    if gpu_host is None:
        gpu_host = shutil.which("nvidia-smi") is not None
    if not gpu_host:
        console.print(
            "[dim]Skipping `gitm install`: no nvidia-smi on this host, so there is no "
            "CUDA stack to prepare. Run `rex install` again on the GPU box.[/dim]"
        )
        return None

    console.rule("Preparing the CUDA host (gitm install)")
    command = gitm_command(["install", *(extra_args or [])])
    run_command(command)
    console.log("[green]CUDA host prepared[/green]")
    return command


def install_runtime(
    dest=None,
    url=RUNTIME_REPO_URL,
    ref=None,
    extras=None,
    editable=True,
    constraints=True,
    update=True,
    gitm_install=None,
    gitm_install_args=None,
):
    """Clone (or update) the runtime repo, install it, and prepare the CUDA host."""
    repo = clone_or_update(dest=dest, url=url, ref=ref, update=update)
    if extras is None:
        extras = default_extras()
    install_dependencies(repo, extras=extras, editable=editable, constraints=constraints)

    if gitm_install is not False:
        # gpu_host=None lets it decide from nvidia-smi; gitm_install=True forces it
        # even where that probe says no.
        run_gitm_install(gitm_install_args, gpu_host=True if gitm_install else None)

    console.print(f"[bold]Runtime checkout:[/bold] {repo.resolve()}")
    console.print(f"[bold]Extras installed:[/bold] {', '.join(extras) if extras else '(none)'}")
    return repo, extras
