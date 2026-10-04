"""llama.cpp backend for GGUF models. Heavy import happens only inside methods."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..chat import ChatChunk, ChatResponse
from ..exceptions import BackendError, ModelLoadError
from .base import Backend


class LlamaCppBackend(Backend):
    name = "llama_cpp"

    def __init__(self) -> None:
        self._llm: Any = None
        self._model_id = "?"

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import llama_cpp  # noqa: F401
        except ModuleNotFoundError:
            return False, 'Package "llama-cpp-python" is missing. Install it with: pip install "compiwer[llama]"'
        return True, "llama-cpp-python is installed."

    def _pick_gguf(self, model_dir: Path, filename: str | None = None) -> Path:
        if filename:
            p = model_dir / filename
            if not p.exists():
                raise ModelLoadError(f"GGUF file not found: {p}")
            return p
        ggufs = sorted(model_dir.rglob("*.gguf"), key=lambda p: p.stat().st_size)
        if not ggufs:
            raise ModelLoadError(f"No .gguf file in {model_dir}.",
                                 fix="Download a GGUF build: compiwer models download mtrini-svl-1.0")
        return ggufs[len(ggufs) // 2]

    def load(self, model_dir: Path, filename: str | None = None, n_ctx: int = 8192,
             n_gpu_layers: int = -1, verbose: bool = False, **kwargs: Any) -> None:
        self.require()
        from llama_cpp import Llama

        gguf = self._pick_gguf(model_dir, filename)
        try:
            self._llm = Llama(model_path=str(gguf), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                              verbose=verbose)
        except Exception as e:
            raise ModelLoadError(f"llama.cpp failed to load {gguf.name}: {e}",
                                 cause="Incompatible file, too little RAM/VRAM, or a bad download.",
                                 fix="Check integrity with: compiwer models info <id>") from e
        self._model_id = gguf.stem

    def unload(self) -> None:
        self._llm = None

    def _ensure(self) -> Any:
        if self._llm is None:
            raise BackendError("Model is not loaded.", fix="Call load() with a model directory first.")
        return self._llm

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 512,
             temperature: float = 0.7, **kwargs: Any) -> ChatResponse:
        llm = self._ensure()
        try:
            out = llm.create_chat_completion(messages=messages, max_tokens=max_tokens,
                                             temperature=temperature, stream=False)
        except Exception as e:
            raise BackendError(f"Inference failed: {e}") from e
        choice = out["choices"][0]
        usage = out.get("usage", {})
        return ChatResponse(text=choice["message"].get("content") or "", model=self._model_id,
                            usage=dict(usage), finish_reason=choice.get("finish_reason"))

    def chat_stream(self, messages: list[dict[str, str]], max_tokens: int = 512,
                    temperature: float = 0.7, **kwargs: Any) -> Iterator[ChatChunk]:
        llm = self._ensure()
        try:
            for part in llm.create_chat_completion(messages=messages, max_tokens=max_tokens,
                                                   temperature=temperature, stream=True):
                delta = part["choices"][0].get("delta", {}).get("content") or ""
                if delta:
                    yield ChatChunk(delta)
        except Exception as e:
            raise BackendError(f"Streaming failed: {e}") from e
        yield ChatChunk("", done=True)
