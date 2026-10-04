"""OpenAI-compatible request/response schemas (subset)."""
from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover
    raise ImportError('The local API needs FastAPI/Pydantic: pip install "compiwer[server]"')


class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    max_tokens: int = 512
    temperature: float = 0.7


class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str = ""
    stream: bool = False
    max_tokens: int = 256
    temperature: float = 0.7


def sse_data(payload: dict[str, Any]) -> str:
    import json

    return f"data: {json.dumps(payload)}\n\n"
