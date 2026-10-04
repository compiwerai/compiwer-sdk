"""API application factory (stable import path for the future TS SDK too)."""
from __future__ import annotations

try:
    from fastapi import FastAPI
except ModuleNotFoundError:  # pragma: no cover
    raise ImportError('The local API needs FastAPI: pip install "compiwer[server]"')

from .routes import build


def create_app() -> FastAPI:
    return build()


__all__ = ["create_app"]
