"""Streaming chunk types."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatChunk:
    text: str
    done: bool = False
