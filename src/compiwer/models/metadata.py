"""Model metadata helpers (sizes, formats) used by manager/CLI output."""
from __future__ import annotations

from pathlib import Path


def dir_size(path: Path) -> int:
    total = 0
    if path.is_file():
        return path.stat().st_size
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def kind_of(path: Path) -> str:
    names = [p.suffix.lower() for p in path.rglob("*") if p.is_file()][:50]
    if ".gguf" in names:
        return "GGUF"
    if ".safetensors" in names:
        return "safetensors"
    if ".bin" in names:
        return "pytorch"
    return "snapshot"
