"""The `models` facade: list/search/download/info/delete/exists."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..exceptions import ModelDownloadError, ModelNotFoundError
from ..utils import logging as _log
from ..utils.paths import models_dir
from . import downloader
from .metadata import dir_size, fmt_size, kind_of
from .registry import all_models, resolve


def list() -> list[dict[str, Any]]:
    rows = []
    for meta in all_models():
        d = models_dir() / meta.repo.replace("/", "__")
        rows.append({"id": meta.id, "repo": meta.repo, "backend": meta.backend,
                     "installed": d.exists(), "path": str(d) if d.exists() else None,
                     "size": fmt_size(dir_size(d)) if d.exists() else None,
                     "note": meta.note})
    return rows


def search(query: str, limit: int = 20) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError:
        return [{"id": m.id, "repo": m.repo} for m in all_models() if query.lower() in m.id]
    try:
        api = HfApi(token=_log.hf_token() or None)
        out = []
        for m in api.list_models(search=query, limit=limit):
            out.append({"id": m.id, "likes": getattr(m, "likes", 0),
                        "tags": getattr(m, "tags", [])[:8]})
        return out
    except Exception as e:
        raise ModelDownloadError(f"Search failed: {e}", fix="Check your connection (public search needs no token).") from e


def download(model_id: str, *, filename: str | None = None, force: bool = False) -> Path:
    meta = resolve(model_id)
    return downloader.download(meta.repo, filename=filename, force=force)


def info(model_id: str) -> dict[str, Any]:
    meta = resolve(model_id)
    d = models_dir() / meta.repo.replace("/", "__")
    out = {"id": meta.id, "repo": meta.repo, "backend": meta.backend,
           "installed": d.exists(), "note": meta.note, "aliases": meta.aliases}
    if d.exists():
        out.update({"path": str(d), "size": fmt_size(dir_size(d)), "format": kind_of(d)})
    else:
        try:
            out["hub"] = downloader.info_dict(meta.repo)
        except ModelDownloadError as e:
            out["hub_error"] = str(e)[:200]
    return out


def delete(model_id: str) -> bool:
    meta = resolve(model_id)
    d = models_dir() / meta.repo.replace("/", "__")
    if not d.exists():
        raise ModelNotFoundError(f"{meta.id} is not downloaded.", fix=f"Download it with: compiwer models download {meta.id}")
    shutil.rmtree(d)  # explicit user request only — never automatic
    return True


def exists(model_id: str) -> bool:
    try:
        meta = resolve(model_id)
    except ModelNotFoundError:
        return False
    d = models_dir() / meta.repo.replace("/", "__")
    return d.exists() and any(d.iterdir())
