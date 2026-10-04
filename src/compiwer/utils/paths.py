"""Filesystem locations. Override root with COMPIWER_HOME."""
from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    return Path(os.environ.get("COMPIWER_HOME", str(Path.home() / ".compiwer"))).expanduser()


def models_dir() -> Path:
    custom = os.environ.get("COMPIWER_MODELS_DIR")
    d = Path(custom).expanduser() if custom else home() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_file() -> Path:
    return home() / "config.toml"


def pid_file() -> Path:
    return home() / "runtime.pid"


def log_file() -> Path:
    return home() / "compiwer.log"
