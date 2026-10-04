"""Registry: resolve ids/aliases/repos to metadata without hardcoding call sites."""
from __future__ import annotations

from .model import BUILTIN, ModelMetadata, all_models, find


def resolve(model_id: str) -> ModelMetadata:
    meta = find(model_id)
    if meta is not None:
        return meta
    if "/" in model_id:
        repo = model_id.strip()
        backend = "llama_cpp" if repo.lower().endswith("gguf") or "gguf" in repo.lower() else "transformers"
        return ModelMetadata(model_id.split("/")[-1].lower(), repo, "llm", backend,
                             note="Ad-hoc repo (not a curated Mtrini entry).")
    from ..exceptions import ModelNotFoundError

    known = ", ".join(m.id for m in BUILTIN)
    raise ModelNotFoundError(
        f"Unknown model {model_id!r}.",
        cause="No registry entry, alias, or repo path matched.",
        fix=f"Try a known id ({known}), an alias, or a full repo like CompiwerAI/Mtrini-SVL-1.0-GGUF.",
    )


__all__ = ["BUILTIN", "ModelMetadata", "all_models", "find", "resolve"]
