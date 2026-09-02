import pytest

from runtime_harness import __version__, cli
from runtime_harness.experiments import BUNDLED_CONFIG


def test_bundled_config_is_packaged():
    assert BUNDLED_CONFIG.exists(), "starter manifest must ship inside the wheel"


def test_version_flag_reports_package_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_subcommand_is_required():
    with pytest.raises(SystemExit):
        cli.main([])


def test_init_writes_manifest(tmp_path):
    destination = tmp_path / "experiments.yaml"
    assert cli.main(["init", str(destination)]) == 0
    assert "experiments:" in destination.read_text()


def test_init_refuses_to_clobber_without_force(tmp_path):
    destination = tmp_path / "experiments.yaml"
    destination.write_text("keep me")
    assert cli.main(["init", str(destination)]) == 1
    assert destination.read_text() == "keep me"

    assert cli.main(["init", str(destination), "--force"]) == 0
    assert "experiments:" in destination.read_text()


def test_check_surfaces_failure_as_exit_code(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_preflight", boom)
    assert cli.main(["check"]) == 1


def test_check_forwards_flags_to_preflight(monkeypatch, tmp_path):
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return (8, True, "topo", kwargs["hf_home"], kwargs["experiments_dir"])

    monkeypatch.setattr(cli, "run_preflight", spy)
    exit_code = cli.main(
        ["check", "--no-install", "--hf-home", str(tmp_path / "hf"), "--experiments-dir", str(tmp_path / "exp")]
    )

    assert exit_code == 0
    assert seen["install"] is False
    assert seen["hf_home"] == tmp_path / "hf"
    assert seen["experiments_dir"] == tmp_path / "exp"


def test_check_installs_by_default(monkeypatch):
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return (8, True, "topo", None, None)

    monkeypatch.setattr(cli, "run_preflight", spy)
    assert cli.main(["check"]) == 0
    assert seen["install"] is True


def test_run_fails_cleanly_when_manifest_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["run"]) == 1


# --- sweep expansion --------------------------------------------------------


def test_variants_expand_into_one_experiment_each(tmp_path):
    """A sweep is written once and expanded, not copy-pasted N times. Duplicated
    blocks drift apart on the fields nobody meant to vary, and a sweep whose arms
    differ in more than the thing under test measures nothing."""
    from runtime_harness.experiments import load_experiments

    config = tmp_path / "e.yaml"
    config.write_text(
        "gpu_counts: [1]\n"
        "experiments:\n"
        "  - name: base\n"
        "    model: m\n"
        "    prompt: p\n"
        "    vllm_args: ['--shared']\n"
        "    variants:\n"
        "      - name: a\n"
        "        vllm_args: ['--only-a']\n"
        "      - name: b\n"
        "        vllm_args: ['--only-b']\n",
        encoding="utf-8",
    )

    _, experiments = load_experiments(config)

    assert [e.name for e in experiments] == ["base-a", "base-b"]
    # base flags reach every arm; variant flags reach only their own
    assert experiments[0].vllm_args == ["--shared", "--only-a"]
    assert experiments[1].vllm_args == ["--shared", "--only-b"]
    assert experiments[0].metadata["variant"] == "a"


def test_an_entry_without_variants_is_unchanged(tmp_path):
    from runtime_harness.experiments import load_experiments

    config = tmp_path / "e.yaml"
    config.write_text(
        "gpu_counts: [1]\n"
        "experiments:\n"
        "  - name: solo\n"
        "    model: m\n"
        "    prompt: p\n"
        "    vllm_args: ['--x']\n",
        encoding="utf-8",
    )

    _, experiments = load_experiments(config)

    assert [e.name for e in experiments] == ["solo"]
    assert experiments[0].vllm_args == ["--x"]


def test_each_variant_writes_its_own_guidellm_report():
    """The report path is baked into a --output argument. Shared across arms, every
    arm overwrites the last and the run directory still looks complete — holding one
    arm's numbers under eight arms' names."""
    from runtime_harness.experiments import retarget_output_path

    command = ["--output", "kind=json,path=/w/run.json"]

    a = retarget_output_path(command, "eager")
    b = retarget_output_path(command, "full")

    assert a == ["--output", "kind=json,path=/w/run-eager.json"]
    assert b == ["--output", "kind=json,path=/w/run-full.json"]
    assert a != b


def test_variant_without_a_name_is_rejected():
    """Unnamed variants would collide on both experiment id and report path, and
    the collision is silent — one arm's result under another arm's name."""
    import pytest

    from runtime_harness.experiments import expand_variants

    with pytest.raises(ValueError, match="needs a name"):
        expand_variants({"name": "base", "variants": [{"vllm_args": ["--x"]}]})
