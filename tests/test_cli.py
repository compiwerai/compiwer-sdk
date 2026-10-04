"""CLI: help, version, hardware, doctor, models commands (offline)."""
from typer.testing import CliRunner

from compiwer.cli.main import app

runner = CliRunner()


def test_help():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0 and "models" in r.output and "serve" in r.output


def test_version():
    assert "compiwer 0.1.0" in runner.invoke(app, ["version"]).output


def test_hardware(home):
    out = runner.invoke(app, ["hardware"]).output
    assert "CPU" in out and "GPU" in out


def test_doctor_offline(home, monkeypatch):
    monkeypatch.setenv("COMPIWER_HOME", str(home))
    out = runner.invoke(app, ["doctor"]).output
    assert "Compiwer Doctor" in out and "Backend llama_cpp" in out


def test_models_commands(home, gguf_dir):
    assert "mtrini-svl-1.0" in runner.invoke(app, ["models", "list"]).output
    assert "GGUF" in runner.invoke(app, ["models", "info", "mtrini-svl-1.0"]).output
    r = runner.invoke(app, ["models", "delete", "mtrini-svl-1.0"], input="n\n")
    assert r.exit_code != 0  # declined
    assert "Deleted" in runner.invoke(app, ["models", "delete", "mtrini-svl-1.0", "--yes"]).output
