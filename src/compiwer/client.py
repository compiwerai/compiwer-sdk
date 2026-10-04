"""Developer-facing API: Model, Agent, ImageModel, Tool, models, hardware."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config
from .backends import detect_backend
from .backends import get as backend_get
from .backends.base import Backend
from .chat import ChatChunk, ChatResponse, normalize_messages
from .exceptions import BackendUnavailableError, CompiwerError, ModelLoadError
from .models import manager as _models
from .models.registry import resolve
from .utils import hardware as _hardware
from .utils.paths import models_dir


class Model:
    """Load a local model and chat with it.

    model = Model("Mtrini-SVL-1.0")          # backend auto-detected
    model = Model("Mtrini-SVL-1.0", backend="llama_cpp")
    model.chat("Write a Python calculator.")
    """

    def __init__(self, model_id: str, backend: str = "auto", **load_kwargs: Any):
        self.meta = resolve(model_id)
        self._backend_name = backend
        self._load_kwargs = load_kwargs
        self._backend: Backend | None = None

    @property
    def id(self) -> str:
        return self.meta.id

    def _model_dir(self) -> Path:
        d = models_dir() / self.meta.repo.replace("/", "__")
        if not d.exists() or not any(d.iterdir()):
            raise ModelLoadError(
                f"{self.meta.id} is not downloaded.",
                fix=f"Download it first: compiwer models download {self.meta.id}")
        return d

    def _ensure(self) -> Backend:
        if self._backend is not None:
            return self._backend
        d = self._model_dir()
        name = detect_backend(d) if self._backend_name == "auto" else self._backend_name
        cls = backend_get(name)
        cls.require()
        be = cls()
        be.load(d, **self._load_kwargs)
        self._backend = be
        self._backend_name = name
        return be

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def chat(self, prompt: str | list[dict[str, str]], **kwargs: Any) -> ChatResponse:
        be = self._ensure()
        resp = be.chat(normalize_messages(prompt), **kwargs)
        resp.model = self.meta.id
        return resp

    def chat_stream(self, prompt: str | list[dict[str, str]], **kwargs: Any) -> Iterator[ChatChunk]:
        yield from self._ensure().chat_stream(normalize_messages(prompt), **kwargs)

    def generate(self, prompt: str, **kwargs: Any) -> ChatResponse:
        return self.chat(prompt, **kwargs)

    def unload(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None


@dataclass
class Tool:
    name: str
    description: str
    dangerous: bool = False
    handler: Callable[[dict[str, Any]], Any] | None = None

    def run(self, args: dict[str, Any]) -> Any:
        assert self.handler is not None
        return self.handler(args)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)


def _read_file(args: dict[str, Any]) -> str:
    p = Path(args["path"]).expanduser()
    if not p.is_file():
        raise CompiwerError(f"Not a file: {p}")
    data = p.read_bytes()
    if len(data) > 200_000:
        raise CompiwerError("File too large for the agent (>200KB).")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise CompiwerError("Binary file — refusing to dump it into context.")


def _list_dir(args: dict[str, Any]) -> list[str]:
    p = Path(args.get("path", ".")).expanduser()
    return sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())[:200]


def default_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool("read_file", "Read a text file (<=200KB). Args: {path}.", False, _read_file))
    reg.register(Tool("list_dir", "List a directory. Args: {path?}.", False, _list_dir))
    return reg


_TOOL_CALL_RE = re.compile(r"```tool\s+(\{.*?\})\s*```", re.DOTALL)


class Agent:
    """Tiny ReAct-style loop over safe tools. No shell, no deletes in v1.

    agent = Agent(model="Mtrini-SVL-1.0")
    agent.run("List the python files here and summarize them.")
    """

    SYSTEM = ("You can call tools by emitting exactly one fenced block per turn:\n"
              "```tool {\"name\": \"read_file\", \"args\": {\"path\": \"x\"}}```\n"
              "Available tools are listed in the user message. Otherwise answer directly.")

    def __init__(self, model: str, backend: str = "auto", max_steps: int = 6,
                 tools: ToolRegistry | None = None, allow_dangerous: bool = False, **kw: Any):
        self.model = Model(model, backend=backend, **kw)
        self.max_steps = max_steps
        self.tools = tools or default_tools()
        self.allow_dangerous = allow_dangerous

    def run(self, task: str) -> str:
        names = ", ".join(f"{n} ({t.description})" for n, t in
                          ((n, self.tools.get(n)) for n in self.tools.names()))
        history = [{"role": "system", "content": self.SYSTEM},
                   {"role": "user", "content": f"Tools: {names}\n\nTask: {task}"}]
        out = ""
        for _ in range(self.max_steps):
            out = self.model.chat(history).text
            m = _TOOL_CALL_RE.search(out)
            if not m:
                return out.strip()
            try:
                call = json.loads(m.group(1))
                tool = self.tools.get(call.get("name", ""))
                if tool is None:
                    history.append({"role": "user", "content": f"Unknown tool {call.get('name')!r}."})
                    continue
                if tool.dangerous and not self.allow_dangerous:
                    history.append({"role": "user",
                                    "content": f"Tool {tool.name} is disabled (dangerous). Answer without it."})
                    continue
                result = tool.run(call.get("args", {}))
                history.append({"role": "user", "content": f"[{tool.name}] result:\n{result}"[:6000]})
            except CompiwerError as e:
                history.append({"role": "user", "content": f"Tool error: {e.message}"})
            except (ValueError, KeyError) as e:
                history.append({"role": "user", "content": f"Bad tool call: {e}. Use the exact JSON format."})
        return out.strip()


class ImageModel:
    """Future Mtrini Imagine entry point. Honest until a backend lands."""

    def __init__(self, model_id: str = "mtrini-imagine-1.0-7b"):
        self.meta = resolve(model_id)

    def generate(self, prompt: str, **kwargs: Any) -> Path:
        raise BackendUnavailableError(
            f"Image generation for {self.meta.id} is not implemented in SDK v1.",
            cause="No image backend (e.g. diffusers) is bundled or registered.",
            fix="Track Mtrini-Imagine support in the roadmap; the ImageBackend interface is ready for it.",
        )

    def edit(self, image: Path, prompt: str, **kwargs: Any) -> Path:
        raise BackendUnavailableError("Image editing is not implemented in SDK v1.")


models = _models
hardware = _hardware

__all__ = ["Agent", "ImageModel", "Model", "Tool", "ToolRegistry", "default_tools",
           "hardware", "models", "config"]
