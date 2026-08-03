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
