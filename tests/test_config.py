"""Config: file + env, secret refusal, masking."""
from compiwer import config


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("COMPIWER_HOME", str(tmp_path))
    from compiwer.utils.paths import home

    assert home() == tmp_path


def test_secret_refusal(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPIWER_HOME", str(tmp_path))
    try:
        config.save({"HF_TOKEN": "x"})
        raise AssertionError("should have refused")
    except Exception as e:
        assert "secret" in str(e).lower()


def test_save_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPIWER_HOME", str(tmp_path))
    p = config.save({"default_backend": "auto", "port": "8000"})
    assert p.exists() and config.get("default_backend") == "auto"


def test_masked():
    out = config.masked({"a": "b", "HF_TOKEN": "x"})
    assert out["a"] == "b"
