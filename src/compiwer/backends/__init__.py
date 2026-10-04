"""Backend registry: name -> implementation for auto-selection and errors."""
from __future__ import annotations

from .base import Backend, detect_backend
from .image import ImageBackend
from .llama_cpp import LlamaCppBackend
from .transformers import TransformersBackend

BACKENDS: dict[str, type[Backend]] = {
    "llama_cpp": LlamaCppBackend,
    "transformers": TransformersBackend,
}


def get(name: str) -> type[Backend]:
    if name == "auto":
        raise ValueError("Use detect_backend() for auto selection.")
    try:
        return BACKENDS[name]
    except KeyError:
        from ..exceptions import BackendUnavailableError

        raise BackendUnavailableError(
            f"Unknown backend {name!r}.",
            fix=f"Available: {', '.join(sorted(BACKENDS))} (plus 'auto').") from None


__all__ = ["BACKENDS", "Backend", "ImageBackend", "LlamaCppBackend", "TransformersBackend",
           "detect_backend", "get"]
