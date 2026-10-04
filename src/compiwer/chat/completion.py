"""Chat completion types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatResponse:
    text: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None


def normalize_messages(prompt: str | list[dict[str, str]]) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    out = []
    for m in prompt:
        out.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    return out
