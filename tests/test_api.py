"""API routes against a stubbed Model (no weights needed)."""
import pytest


@pytest.fixture()
def client(monkeypatch, home, gguf_dir):
    from fastapi.testclient import TestClient

    import compiwer.client as _client
    from compiwer.api.app import create_app

    real_model = _client.Model

    class StubModel(real_model):
        def _ensure(self):  # skip backend entirely
            from tests.conftest import MockBackend

            self._backend = MockBackend()
            return self._backend

    monkeypatch.setattr(_client, "Model", StubModel)
    monkeypatch.setattr("compiwer.api.routes.Model", StubModel)
    return TestClient(create_app())


def test_health_and_models(client):
    assert client.get("/health").json()["status"] == "ok"
    data = client.get("/v1/models").json()["data"]
    assert any(m["id"] == "mtrini-svl-1.0" and m["installed"] for m in data)


def test_chat_completions(client):
    r = client.post("/v1/chat/completions", json={
        "model": "mtrini-svl-1.0", "messages": [{"role": "user", "content": "Hello!"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"].startswith("mock:")


def test_chat_stream(client):
    r = client.post("/v1/chat/completions", json={
        "model": "mtrini-svl-1.0", "messages": [{"role": "user", "content": "Hi"}], "stream": True})
    assert r.status_code == 200 and "mock:" in r.text and "[DONE]" in r.text


def test_completions(client):
    r = client.post("/v1/completions", json={"model": "mtrini-svl-1.0", "prompt": "Hi"})
    assert r.status_code == 200
    assert r.json()["choices"][0]["text"].startswith("mock:")


def test_empty_messages_rejected(client):
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 400
