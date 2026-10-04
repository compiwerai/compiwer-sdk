"""Image backend extension point (future Mtrini Imagine support).

No image engine is bundled in v1 — this interface keeps the door open without
pretending generation works. `ImageModel.generate()` raises an honest error
until a backend (e.g. diffusers) is installed and registered.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ImageBackend(ABC):
    name: str = "image-base"

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        return False, "No image backend is installed yet."

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> Path:
        """Render prompt to an image file."""

    @abstractmethod
    def edit(self, image: Path, prompt: str, **kwargs: Any) -> Path:
        """Edit an image by instruction."""
