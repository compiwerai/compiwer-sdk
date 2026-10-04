"""Backends: selection, availability errors, mock inference, agent loop, image honesty."""
import pytest

from compiwer.backends import get as backend_get
from compiwer.backends.base import detect_backend
from compiwer.client import Agent, ImageModel, Model
from compiwer.exceptions import BackendUnavailableError, ModelLoadError
from tests.conftest import MockBackend


def test_unknown_backend():
    with pytest.raises(BackendUnavailableError):
        backend_get("nope")


def test_detect_gguf(gguf_dir):
    assert detect_backend(gguf_dir) == "llama_cpp"


def test_detect_empty(tmp_path):
    with pytest.raises(ModelLoadError):
        detect_backend(tmp_path)


def test_model_uses_mock_backend(home, gguf_dir, monkeypatch):
    import compiwer.client as _client

    monkeypatch.setattr(_client, "backend_get", lambda name: MockBackend)
    m = Model("mtrini-svl-1.0")
    resp = m.chat("Hello!")
    assert resp.text.startswith("mock:") and resp.model == "mtrini-svl-1.0"
    assert list(m.chat_stream("Hi"))[-1].done is True


def test_model_missing_model(home):
    with pytest.raises(ModelLoadError) as e:
        Model("mtrini-tellus-27b").chat("hi")
    assert "compiwer models download" in str(e.value)


def test_agent_readonly_loop(home, gguf_dir, monkeypatch, tmp_path):
    import compiwer.client as _client

    (tmp_path / "a.txt").write_text("hello agent")
    monkeypatch.setattr(_client, "backend_get", lambda name: MockBackend)
    monkeypatch.chdir(tmp_path)
    out = Agent(model="mtrini-svl-1.0", max_steps=1).run("do something")
    assert isinstance(out, str)


def test_agent_tool_direct(tmp_path):
    from compiwer.client import default_tools

    (tmp_path / "x.txt").write_text("data")
    reg = default_tools()
    assert reg.get("read_file").run({"path": str(tmp_path / "x.txt")}) == "data"
    assert "y.txt" not in str(reg.get("list_dir").run({"path": str(tmp_path)}))


def test_image_model_honest():
    with pytest.raises(BackendUnavailableError) as e:
        ImageModel("mtrini-imagine-1.0-7b").generate("A futuristic Moroccan city")
    assert "not implemented" in str(e.value)
