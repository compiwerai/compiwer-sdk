"""Backend abstraction. Engines plug in here; the SDK never invents inference."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..chat import ChatChunk, ChatResponse


class Backend(ABC):
    """One local inference engine behind a stable interface."""

    name: str = "base"

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """(available, human-readable reason / install hint). Never raises, never imports heavy deps."""
        return False, f"{cls.name} is not implemented."

    @classmethod
    def require(cls) -> None:
        ok, reason = cls.is_available()
        if not ok:
            from ..exceptions import BackendUnavailableError

            raise BackendUnavailableError(f"Backend {cls.name!r} is unavailable.", cause=reason)

    @abstractmethod
    def load(self, model_dir: Path, **kwargs: Any) -> None:
        """Load weights into memory."""

    @abstractmethod
    def unload(self) -> None:
        """Free the model."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatResponse:
        """Single completion."""

    @abstractmethod
    def chat_stream(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[ChatChunk]:
        """Streamed completion."""

    def generate(self, prompt: str, **kwargs: Any) -> ChatResponse:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def capabilities(self) -> dict[str, Any]:
        return {"chat": True, "stream": True, "tools": False, "vision": False}


def detect_backend(model_dir: Path) -> str:
    """Pick an engine from files on disk: .gguf -> llama_cpp, safetensors/config -> transformers."""
    try:
        files = [p.suffix.lower() for p in model_dir.rglob("*") if p.is_file()]
    except OSError:
        files = []
    if ".gguf" in files:
        return "llama_cpp"
    if ".safetensors" in files or ".bin" in files:
        return "transformers"
    from ..exceptions import ModelLoadError

    raise ModelLoadError(
        f"No runnable weights in {model_dir}.",
        cause="No .gguf, .safetensors, or .bin files found.",
        fix="Download a runnable build, e.g. compiwer models download mtrini-svl-1.0",
    )
