"""Compiwer SDK errors — every failure explains what happened and what to try next."""
from __future__ import annotations


class CompiwerError(Exception):
    """Base error. Carries a human message plus optional cause and fix hint."""

    def __init__(self, message: str, *, cause: str = "", fix: str = ""):
        self.message = message
        self.cause = cause
        self.fix = fix
        super().__init__(str(self))

    def __str__(self) -> str:
        parts = [self.message]
        if self.cause:
            parts.append(f"\nPossible cause:\n{self.cause}")
        if self.fix:
            parts.append(f"\nTry:\n{self.fix}")
        return "\n".join(parts)


class ModelNotFoundError(CompiwerError):
    """Unknown model id and nothing like it on disk."""


class ModelDownloadError(CompiwerError):
    """Download failed (network, auth, disk space, corrupt file)."""


class BackendError(CompiwerError):
    """Backend failed at runtime (load/inference crashed)."""


class BackendUnavailableError(CompiwerError):
    """Backend cannot run here (missing dependency, no file, no GPU build)."""


class ModelLoadError(CompiwerError):
    """Model files present but loading failed."""


class ConfigurationError(CompiwerError):
    """Bad config file or environment."""


class RuntimeError(CompiwerError):
    """Local server/runtime failed to start or answer."""
