"""Shared fixtures: isolated COMPIWER_HOME + a fake GGUF tree + mock backend."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from compiwer.backends.base import Backend
from compiwer.chat import ChatChunk, ChatResponse


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "compiwer-home"
    monkeypatch.setenv("COMPIWER_HOME", str(h))
    return h


@pytest.fixture()
def gguf_dir(home: Path) -> Path:
    d = home / "models" / "CompiwerAI__Mtrini-SVL-1.0-GGUF"
    d.mkdir(parents=True)
    (d / "model-q4_k_m.gguf").write_bytes(b"GGUF" * 256)
    return d


class MockBackend(Backend):
    name = "mock-test"

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.loaded: Path | None = None

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        return True, "test double"

    def load(self, model_dir: Path, **kwargs: Any) -> None:
        self.loaded = model_dir

    def unload(self) -> None:
        self.loaded = None

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatResponse:
        if self.fail:
            from compiwer.exceptions import BackendError

            raise BackendError("mock failure")
        last = messages[-1]["content"] if messages else ""
        return ChatResponse(f"mock:{last[:20]}", "mock-model",
                            {"prompt_tokens": 1, "completion_tokens": 1}, "stop")

    def chat_stream(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[ChatChunk]:
        yield ChatChunk("mock:")
        yield ChatChunk("hi", done=False)
        yield ChatChunk("", done=True)
