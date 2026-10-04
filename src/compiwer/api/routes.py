"""Local OpenAI-compatible API: /v1/chat/completions, /v1/completions, /v1/models, /health."""
from __future__ import annotations

import time
import uuid
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
except ModuleNotFoundError:  # pragma: no cover
    raise ImportError('The local API needs FastAPI: pip install "compiwer[server]"')

from ..client import Model
from ..exceptions import CompiwerError
from ..models import manager as models
from ..utils import hardware as _hw
from . import schemas as S


def _get_model(name: str) -> Model:
    try:
        return Model(name or "mtrini-svl-1.0")
    except CompiwerError as e:
        raise HTTPException(503, str(e))


def _chat_to_openai(resp: Any, model: str, stream: bool = False) -> dict[str, Any]:
    base: dict[str, Any] = {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
                            "created": int(time.time()), "model": model}
    if stream:
        base["object"] = "chat.completion.chunk"
        text = resp.text if hasattr(resp, "text") else resp
        base["choices"] = [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
    else:
        base["choices"] = [{"index": 0, "message": {"role": "assistant", "content": resp.text},
                            "finish_reason": resp.finish_reason or "stop"}]
        base["usage"] = dict(resp.usage or {})
    return base


def create_app() -> FastAPI:
    app = FastAPI(title="Compiwer Runtime", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "runtime": "compiwer-local"}

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        data = [{"id": m["id"], "object": "model", "owned_by": "compiwer",
                 "backend": m["backend"], "installed": m["installed"]} for m in models.list()]
        return {"object": "list", "data": data}

    @app.get("/v1/hardware")
    def hw() -> dict[str, Any]:
        return _hw.info()

    @app.post("/v1/chat/completions")
    def chat_completions(body: S.ChatRequest) -> Any:
        msgs = [{"role": m.role, "content": m.content} for m in body.messages]
        if not msgs:
            raise HTTPException(400, "messages must not be empty")
        model = _get_model(body.model)

        def _run_stream():  # type: ignore[no-untyped-def]
            try:
                for ch in model.chat_stream(msgs, max_tokens=body.max_tokens,
                                            temperature=body.temperature):
                    if ch.text:
                        yield S.sse_data(_chat_to_openai(ch, body.model or model.id, True))
                yield "data: [DONE]\n\n"
            except CompiwerError as e:
                yield S.sse_data({"error": str(e)})

        try:
            if body.stream:
                return StreamingResponse(_run_stream(), media_type="text/event-stream")
            resp = model.chat(msgs, max_tokens=body.max_tokens, temperature=body.temperature)
            return _chat_to_openai(resp, body.model or model.id)
        except CompiwerError as e:
            raise HTTPException(503, str(e))

    @app.post("/v1/completions")
    def completions(body: S.CompletionRequest) -> Any:
        model = _get_model(body.model)

        def _run_stream():  # type: ignore[no-untyped-def]
            try:
                for ch in model.chat_stream(body.prompt, max_tokens=body.max_tokens,
                                            temperature=body.temperature):
                    if ch.text:
                        yield S.sse_data({"id": f"cmpl-{uuid.uuid4().hex[:8]}",
                                          "object": "text_completion",
                                          "choices": [{"text": ch.text, "finish_reason": None}]})
                yield "data: [DONE]\n\n"
            except CompiwerError as e:
                yield S.sse_data({"error": str(e)})

        try:
            if body.stream:
                return StreamingResponse(_run_stream(), media_type="text/event-stream")
            resp = model.generate(body.prompt, max_tokens=body.max_tokens, temperature=body.temperature)
            return {"id": f"cmpl-{uuid.uuid4().hex[:8]}", "object": "text_completion",
                    "model": body.model or model.id,
                    "choices": [{"text": resp.text, "finish_reason": resp.finish_reason or "stop"}]}
        except CompiwerError as e:
            raise HTTPException(503, str(e))

    @app.exception_handler(HTTPException)
    async def _http_exc(_req: Any, exc: HTTPException) -> JSONResponse:  # type: ignore[no-untyped-def]
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    return app


def build() -> FastAPI:
    return create_app()
