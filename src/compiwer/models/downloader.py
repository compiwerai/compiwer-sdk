"""Downloads from Hugging Face Hub: resume, disk checks, auth, integrity. Never deletes user data."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..exceptions import ModelDownloadError
from ..utils import logging as _log
from ..utils.paths import models_dir


def _hub():
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ModuleNotFoundError:
        raise ModelDownloadError(
            "huggingface-hub is not installed.",
            cause="The SDK needs it for downloads.",
            fix='pip install "compiwer[all]"  (or: pip install huggingface-hub)',
        )
    return HfApi, hf_hub_download, snapshot_download


def repo_size_approx(repo: str, token: str | None) -> int | None:
    HfApi, _, _ = _hub()
    try:
        info = HfApi(token=token or None).model_info(repo, files_metadata=True)
        total = sum(getattr(s, "size", 0) or 0 for s in (info.siblings or []))
        return total or None
    except Exception:
        return None


def _target_dir(repo: str) -> Path:
    return models_dir() / repo.replace("/", "__")


def download(repo: str, *, filename: str | None = None, token: str | None = None,
             force: bool = False) -> Path:
    """Download a repo snapshot (or one file). Returns the local directory."""
    HfApi, hf_hub_download, snapshot_download = _hub()
    log = _log.setup()
    tok = token or _log.hf_token()
    dest = _target_dir(repo)
    if dest.exists() and any(dest.iterdir()) and not force:
        log.warning("Already downloaded: %s (use --force to re-fetch)", dest)
        return dest
    approx = repo_size_approx(repo, tok)
    if approx:
        free = shutil.disk_usage(dest.parent).free
        if free < approx * 1.1:
            raise ModelDownloadError(
                f"Not enough disk space for {repo} (~{approx / 1e9:.1f} GB, {free / 1e9:.1f} GB free).",
                cause="Model weights are large.",
                fix="Free disk space or set COMPIWER_MODELS_DIR to a bigger drive.",
            )
    dest.mkdir(parents=True, exist_ok=True)
    try:
        if filename:
            hf_hub_download(repo_id=repo, filename=filename, local_dir=str(dest),
                            token=tok or None, resume_download=True)
        else:
            snapshot_download(repo_id=repo, local_dir=str(dest), token=tok or None,
                              resume_download=True)
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower() or "private" in msg.lower():
            raise ModelDownloadError(
                f"Access denied for {repo}.",
                cause="Private or gated repository.",
                fix="Accept the terms on huggingface.co, then export HF_TOKEN=hf_... and retry.",
            ) from e
        if "404" in msg or "not found" in msg.lower():
            raise ModelDownloadError(f"Repository or file not found: {repo} {filename or ''}.",
                                     fix="Check the id with: compiwer models search <name>") from e
        raise ModelDownloadError(f"Download failed: {msg[:300]}",
                                 cause="Network or hub error; partial files are kept for resume.",
                                 fix="Retry the same command — downloads resume.") from e
    return dest


def info_dict(repo: str, token: str | None = None) -> dict[str, Any]:
    HfApi, _, _ = _hub()
    try:
        info = HfApi(token=token or _log.hf_token() or None).model_info(repo)
        return {"id": info.id, "pipeline": getattr(info, "pipeline_tag", None),
                "tags": getattr(info, "tags", []),
                "files": [s.rfilename for s in getattr(info, "siblings", []) or []]}
    except Exception as e:
        raise ModelDownloadError(f"Cannot read {repo}: {e}", fix="Check the id and your connection.") from e
