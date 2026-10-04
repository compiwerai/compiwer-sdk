"""Transformers backend for safetensors/pytorch models. Heavy imports stay inside methods."""
from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..chat import ChatChunk, ChatResponse
from ..exceptions import BackendError, BackendUnavailableError, ModelLoadError
from .base import Backend


class TransformersBackend(Backend):
    name = "transformers"

    def __init__(self) -> None:
        self._model: Any = None
        self._tok: Any = None
        self._model_id = "?"

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        missing = []
        for pkg in ("transformers", "torch"):
            try:
                __import__(pkg)
            except ModuleNotFoundError:
                missing.append(pkg)
        if missing:
            return False, ("Missing: " + ", ".join(missing) +
                           '. Install with: pip install "compiwer[transformers]" '
                           "(CPU users: install torch from pytorch.org first if needed)")
        return True, "transformers + torch are installed."

    def _device(self, device: str | None) -> str:
        if device:
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def load(self, model_dir: Path, device: str | None = None, dtype: str | None = None,
             trust_remote_code: bool = False, **kwargs: Any) -> None:
        self.require()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as e:
            raise BackendUnavailableError(str(e)) from e
        dev = self._device(device)
        dt = None
        if dtype:
            dt = getattr(torch, dtype, None)
        if dt is None:
            dt = torch.float16 if dev == "cuda" else torch.float32
        try:
            self._tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    str(model_dir), dtype=dt, device_map="auto" if dev == "cuda" else None,
                    trust_remote_code=trust_remote_code)
            except TypeError:  # older transformers: torch_dtype
                self._model = AutoModelForCausalLM.from_pretrained(
                    str(model_dir), torch_dtype=dt, device_map="auto" if dev == "cuda" else None,
                    trust_remote_code=trust_remote_code)
            if dev == "cpu":
                self._model.to("cpu")
            self._model.eval()
        except Exception as e:
            raise ModelLoadError(f"transformers failed to load {model_dir}: {e}",
                                 fix="Adapters need their base model; merges need full weights.") from e
        self._model_id = model_dir.name

    def unload(self) -> None:
        self._model, self._tok = None, None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _prompt(self, messages: list[dict[str, str]]) -> str:
        if self._tok is None:
            raise BackendError("Model is not loaded.")
        try:
            return self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 512,
             temperature: float = 0.7, **kwargs: Any) -> ChatResponse:
        import torch

        if self._model is None or self._tok is None:
            raise BackendError("Model is not loaded.", fix="Call load() first.")
        ids = self._tok(self._prompt(messages), return_tensors="pt").to(self._model.device)
        try:
            with torch.no_grad():
                out = self._model.generate(**ids, max_new_tokens=max_tokens,
                                           temperature=max(temperature, 1e-4),
                                           do_sample=temperature > 0, pad_token_id=self._tok.eos_token_id)
        except Exception as e:
            raise BackendError(f"Inference failed: {e}") from e
        text = self._tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return ChatResponse(text=text, model=self._model_id,
                            usage={"prompt_tokens": int(ids["input_ids"].shape[1])},
                            finish_reason="stop")

    def chat_stream(self, messages: list[dict[str, str]], max_tokens: int = 512,
                    temperature: float = 0.7, **kwargs: Any) -> Iterator[ChatChunk]:
        from transformers import TextIteratorStreamer

        if self._model is None or self._tok is None:
            raise BackendError("Model is not loaded.", fix="Call load() first.")
        ids = self._tok(self._prompt(messages), return_tensors="pt").to(self._model.device)
        streamer = TextIteratorStreamer(self._tok, skip_special_tokens=True)
        t = threading.Thread(target=self._model.generate, kwargs={
            **ids, "max_new_tokens": max_tokens, "temperature": max(temperature, 1e-4),
            "do_sample": temperature > 0, "pad_token_id": self._tok.eos_token_id,
            "streamer": streamer}, daemon=True)
        t.start()
        try:
            for piece in streamer:
                if piece:
                    yield ChatChunk(piece)
        except Exception as e:
            raise BackendError(f"Streaming failed: {e}") from e
        t.join()
        yield ChatChunk("", done=True)
