"""Tests for `rex install`: clone the runtime repo, then install it.

The pip and git calls are stubbed throughout. What matters is the command the
harness *builds* -- a wrong extra or a missing index URL fails minutes into a
multi-gigabyte install on the cluster, not here.
"""

from pathlib import Path

import pytest

from runtime_harness import cli, install


@pytest.fixture
def recorded(monkeypatch):
    """Capture every command install.py would run, without running any."""
    commands = []

    def fake_run(command, capture_output=False, check=True, env=None, cwd=None):
        commands.append([str(part) for part in command])
        return "" if capture_output else None

    monkeypatch.setattr(install, "run_command", fake_run)
    monkeypatch.setattr(install.shutil, "which", lambda name: f"/usr/bin/{name}")
    return commands


def make_checkout(path: Path, url=install.RUNTIME_REPO_URL) -> Path:
    """A directory that looks enough like a clone for install.py's purposes."""
    (path / ".git").mkdir(parents=True)
    (path / "pyproject.toml").write_text("[project]\nname = 'gitm-labs'\n")
    (path / install.CONSTRAINTS_NAME).write_text("numpy==2.2.6\n")
    return path


# --- what gets cloned -------------------------------------------------------


def test_install_clones_the_runtime_repo(recorded, monkeypatch, tmp_path):
    dest = tmp_path / "runtime"

    def fake_run(command, capture_output=False, check=True, env=None, cwd=None):
        recorded.append([str(part) for part in command])
        if command[:2] == ["git", "clone"]:
            make_checkout(dest)
        return "" if capture_output else None

    monkeypatch.setattr(install, "run_command", fake_run)
    install.install_runtime(dest=dest, extras=[])

    clone = next(c for c in recorded if c[:2] == ["git", "clone"])
    assert install.RUNTIME_REPO_URL in clone
    assert str(dest) in clone


def test_a_ref_is_cloned_as_a_branch_and_a_sha_is_checked_out_after(recorded, tmp_path):
    """--branch <sha> is not valid git; a pinned commit needs a second step."""
    branch_dest = tmp_path / "branch"
    install.clone_or_update(dest=branch_dest, ref="main")
    assert ["git", "clone", "--branch", "main", install.RUNTIME_REPO_URL, str(branch_dest)] in recorded

    recorded.clear()
    sha_dest = tmp_path / "sha"
    install.clone_or_update(dest=sha_dest, ref="abe4580")
    assert "--branch" not in recorded[0]
    assert recorded[-1] == ["git", "-C", str(sha_dest), "checkout", "--force", "abe4580"]


def test_existing_checkout_is_updated_not_recloned(recorded, tmp_path):
    """Re-running install is how you pick up a new commit."""
    dest = make_checkout(tmp_path / "runtime")

    install.clone_or_update(dest=dest)

    assert not any(command[:2] == ["git", "clone"] for command in recorded)
    assert ["git", "-C", str(dest), "fetch", "--tags", "origin", "HEAD"] in recorded


def test_no_update_leaves_an_existing_checkout_alone(recorded, tmp_path):
    dest = make_checkout(tmp_path / "runtime")

    install.clone_or_update(dest=dest, update=False)

    assert not any(command[:1] == ["git"] and "fetch" in command for command in recorded)


def test_a_checkout_of_a_different_repo_is_refused(monkeypatch, tmp_path):
    """The destination is never deleted to make room -- a wrong --dest that
    happens to point at someone's working tree must not eat it."""
    dest = make_checkout(tmp_path / "runtime")
    monkeypatch.setattr(install.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        install,
        "run_command",
        lambda *a, **kw: "https://github.com/someone/else.git\n",
    )

    with pytest.raises(RuntimeError, match="not https://github.com/GitM-Labs/runtime.git"):
        install.clone_or_update(dest=dest)


def test_ssh_and_https_remotes_count_as_the_same_repo(recorded, monkeypatch, tmp_path):
    """git@github.com:GitM-Labs/runtime.git and the https URL are one repo; a
    checkout made over SSH must not be rejected as a stranger."""
    dest = make_checkout(tmp_path / "runtime")

    def fake_run(command, capture_output=False, check=True, env=None, cwd=None):
        recorded.append([str(part) for part in command])
        return "git@github.com:GitM-Labs/runtime.git\n" if capture_output else None

    monkeypatch.setattr(install, "run_command", fake_run)
    install.clone_or_update(dest=dest)  # does not raise


def test_a_non_repo_directory_with_files_is_refused(recorded, tmp_path):
    dest = tmp_path / "runtime"
    dest.mkdir()
    (dest / "notes.txt").write_text("mine")

    with pytest.raises(RuntimeError, match="not a git checkout"):
        install.clone_or_update(dest=dest)

    assert (dest / "notes.txt").exists()


# --- what gets installed ----------------------------------------------------


def test_gpu_extra_adds_the_nvidia_index(tmp_path):
    """cuDF and CuPy live on NVIDIA's index, not PyPI. Without the extra index
    the gpu extra cannot resolve at all."""
    repo = make_checkout(tmp_path / "runtime")

    command = install.build_pip_command(repo, extras=["gpu", "vllm"])

    assert "--extra-index-url" in command
    assert install.NVIDIA_INDEX_URL in command
    assert f"{repo.resolve()}[gpu,vllm]" in command


def test_cpu_only_install_skips_the_nvidia_index(tmp_path):
    repo = make_checkout(tmp_path / "runtime")

    command = install.build_pip_command(repo, extras=["bench"])

    assert "--extra-index-url" not in command


def test_constraints_file_is_applied_when_present(tmp_path):
    """The repo's constraints.txt is the version set its suite is green against;
    the same pins here mean the host lands the stack the project tested."""
    repo = make_checkout(tmp_path / "runtime")

    command = install.build_pip_command(repo, extras=["bench"])
    assert command[command.index("-c") + 1] == str((repo / install.CONSTRAINTS_NAME).resolve())

    assert "-c" not in install.build_pip_command(repo, extras=["bench"], constraints=False)


def test_constraints_are_skipped_when_the_repo_has_none(tmp_path):
    repo = make_checkout(tmp_path / "runtime")
    (repo / install.CONSTRAINTS_NAME).unlink()

    assert "-c" not in install.build_pip_command(repo)


def test_editable_by_default_and_plain_with_no_editable(tmp_path):
    repo = make_checkout(tmp_path / "runtime")

    assert "-e" in install.build_pip_command(repo)
    assert "-e" not in install.build_pip_command(repo, editable=False)


def test_install_target_is_an_absolute_path(tmp_path, monkeypatch):
    """`pip install -e runtime[gpu]` is ambiguous with a PyPI project named
    runtime; an absolute path can only be read as a directory."""
    make_checkout(tmp_path / "runtime")
    monkeypatch.chdir(tmp_path)

    command = install.build_pip_command(Path("runtime"), extras=["gpu"])

    target = next(part for part in command if part.endswith("[gpu]"))
    assert target == f"{(tmp_path / 'runtime').resolve()}[gpu]"


def test_an_incomplete_clone_is_reported_before_pip_runs(recorded, tmp_path):
    repo = tmp_path / "runtime"
    repo.mkdir()

    with pytest.raises(RuntimeError, match="no pyproject.toml"):
        install.install_dependencies(repo)

    assert recorded == []


# --- extras selection -------------------------------------------------------


def test_gpu_host_gets_the_gpu_stack_and_a_laptop_does_not(monkeypatch):
    """The gpu extra is Linux/CUDA-only wheels off NVIDIA's index. Defaulting to
    it everywhere would make `rex install` fail on the dev box it was written on."""
    assert install.default_extras(gpu_host=True) == list(install.GPU_HOST_EXTRAS)
    assert install.default_extras(gpu_host=False) == list(install.CPU_HOST_EXTRAS)

    monkeypatch.setattr(install.shutil, "which", lambda name: None)
    assert "gpu" not in install.default_extras()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("gpu,vllm", ["gpu", "vllm"]),
        (" gpu , vllm ", ["gpu", "vllm"]),
        ("gpu vllm", ["gpu", "vllm"]),
        ("gpu,gpu", ["gpu"]),
        ("", []),
        (None, None),
    ],
)
def test_extras_parsing(raw, expected):
    assert install.parse_extras(raw) == expected


def test_empty_extras_means_none_not_the_default(monkeypatch, tmp_path):
    """`--extras ''` is an explicit "base install only"; it must not fall back
    to the host default and drag in the whole CUDA stack."""
    seen = {}
    monkeypatch.setattr(install, "clone_or_update", lambda **kw: make_checkout(tmp_path / "r"))
    monkeypatch.setattr(
        install,
        "install_dependencies",
        lambda repo, **kw: seen.update(kw),
    )

    install.install_runtime(extras=[], gitm_install=False)

    assert seen["extras"] == []


# --- CLI wiring -------------------------------------------------------------


def test_cli_forwards_flags_to_install_runtime(monkeypatch, tmp_path):
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return Path("runtime"), ["gpu"]

    monkeypatch.setattr(cli, "install_runtime", spy)
    exit_code = cli.main(
        [
            "install",
            "--dest", str(tmp_path / "rt"),
            "--ref", "main",
            "--extras", "gpu,vllm",
            "--no-editable",
            "--no-constraints",
            "--no-update",
        ]
    )

    assert exit_code == 0
    assert seen["dest"] == tmp_path / "rt"
    assert seen["url"] == install.RUNTIME_REPO_URL
    assert seen["ref"] == "main"
    assert seen["extras"] == ["gpu", "vllm"]
    assert seen["editable"] is False
    assert seen["constraints"] is False
    assert seen["update"] is False


def test_cli_defaults_leave_extras_to_the_host(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "install_runtime", lambda **kw: (seen.update(kw), (Path("runtime"), []))[1])

    assert cli.main(["install"]) == 0
    assert seen["extras"] is None
    assert seen["editable"] is True
    assert seen["constraints"] is True
    assert seen["update"] is True


def test_cli_reports_a_failed_install_as_a_nonzero_exit(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("git not found in PATH")

    monkeypatch.setattr(cli, "install_runtime", boom)
    assert cli.main(["install"]) == 1


# --- gitm install -----------------------------------------------------------


def test_gitm_install_is_skipped_off_a_cuda_host(recorded, monkeypatch):
    """There is no driver to match and nothing it does applies. Running it anyway
    would try to apt-install build dependencies on a laptop."""
    monkeypatch.setattr(install.shutil, "which", lambda name: None)

    assert install.run_gitm_install() is None
    assert recorded == []


def test_gitm_install_runs_on_a_gpu_host(recorded):
    """This is the step that builds the injection library every capture depends on:
    without it `gitm capture serve` fails preflight at the injection-lib check."""
    command = install.run_gitm_install(["--skip-apt"], gpu_host=True)

    assert command[-2:] == ["install", "--skip-apt"]
    assert recorded[-1] == command


def test_gitm_falls_back_to_this_interpreter(monkeypatch):
    """A `gitm` on PATH from another environment would prepare THAT environment's
    CUDA stack, and nothing in the output would say so."""
    monkeypatch.setattr(install.shutil, "which", lambda name: None)

    command = install.gitm_command(["install"])

    assert command[0] == install.sys.executable
    assert command[1:] == ["-m", "gitm.cli", "install"]
