"""Compiwer SDK — Build AI locally."""
from __future__ import annotations

from . import config
from .client import Agent, ImageModel, Model, Tool, ToolRegistry, default_tools, hardware, models
from .exceptions import (
                         BackendError,
                         BackendUnavailableError,
                         CompiwerError,
                         ConfigurationError,
                         ModelDownloadError,
                         ModelLoadError,
                         ModelNotFoundError,
                         RuntimeError,
)

__version__ = "0.1.0"

__all__ = ["Agent", "BackendError", "BackendUnavailableError", "CompiwerError",
           "ConfigurationError", "ImageModel", "Model", "ModelDownloadError",
           "ModelLoadError", "ModelNotFoundError", "RuntimeError", "Tool",
           "ToolRegistry", "config", "default_tools", "hardware", "models"]
