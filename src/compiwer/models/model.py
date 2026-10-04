"""Model metadata + registry resolution."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelMetadata:
    id: str
    repo: str
    type: str = "llm"  # llm | image
    backend: str = "llama_cpp"  # llama_cpp | transformers | image
    aliases: list[str] = field(default_factory=list)
    note: str = ""


BUILTIN: list[ModelMetadata] = [
    ModelMetadata("mtrini-svl-1.0", "CompiwerAI/Mtrini-SVL-1.0-GGUF", "llm", "llama_cpp",
                  ["mtrini-svl"], "Vision-language coder, GGUF for local runtimes."),
    ModelMetadata("mtrini-svl-1.1", "CompiwerAI/Mtrini-SVL-1.1-GGUF", "llm", "llama_cpp",
                  ["mtrini-svl-1.1-gguf"], "SVL 1.1 merged weights, GGUF build."),
    ModelMetadata("mtrini-tellus-27b", "CompiwerAI/Mtrini-27B-Tellus-GGUF", "llm", "llama_cpp",
                  ["mtrini-tellus", "mtrini-27b"], "27B flagship for coding/conversation (AR/Darija)."),
    ModelMetadata("mtrini-tellus-12b-sahara-2", "CompiwerAI/Mtrini-Tellus-12B-Sahara-2", "llm", "transformers",
                  ["mtrini-12b", "mtrini-sahara"], "12B adapter; needs its base + PEFT at runtime."),
    ModelMetadata("mtrini-imagine-1.0-7b", "CompiwerAI/Mtrini-Imagine-1.0-7B-GGUF", "image", "image",
                  ["mtrini-imagine", "mtrini-imagine-7b"], "Image model (GGUF); backend lands with diffusers support."),
]


def find(model_id: str) -> ModelMetadata | None:
    key = model_id.strip().lower()
    for m in BUILTIN:
        if key == m.id.lower() or key in [a.lower() for a in m.aliases]:
            return m
    return None


def all_models() -> list[ModelMetadata]:
    return list(BUILTIN)
