# Compiwer SDK

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/compiwerai/compiwer-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/compiwerai/compiwer-sdk/actions)
[![PyPI](https://img.shields.io/badge/pip%20install-compiwer-green)](https://github.com/compiwerai/compiwer-sdk)

**Build AI locally.**

Open-source Python SDK + CLI for running Compiwer/Mtrini AI models on your own machine — by **Compiwer AI** (*Building AI For Everyone*). Apache-2.0.

> No account. No API key for local inference. No cloud required. Your prompts never leave your machine unless you point the SDK at a remote endpoint yourself.

```bash
pip install compiwer
compiwer models download CompiwerAI/Mtrini-SVL-1.0-GGUF
compiwer run Mtrini-SVL-1.0
```

```python
from compiwer import Model

model = Model("Mtrini-SVL-1.0")
print(model.chat("Write a Python calculator.").text)

for chunk in model.chat_stream("Explain Python"):
    print(chunk.text, end="")
```

## Why local AI

- **Private** — weights, prompts, and files stay on your hardware.
- **Free** — no per-token bills after the one-time download.
- **Offline** — plane, lab, or air-gapped server.
- **Yours** — Apache-2.0, standard formats (GGUF/safetensors), OpenAI-compatible API.

## Install

```bash
pip install compiwer                          # core (chat, models, CLI, agent)
pip install "compiwer[llama]"                 # llama.cpp / GGUF runtime
pip install "compiwer[transformers]"          # HF transformers runtime (needs torch)
pip install "compiwer[server]"                # local OpenAI-compatible API
pip install "compiwer[all]"                   # everything (development)
```

Python 3.11+. For GPU llama.cpp builds, see [docs/backends.md](docs/backends.md).

## Quick start

```bash
compiwer doctor                               # diagnose your machine
compiwer hardware                             # CPU/RAM/GPU report
compiwer models list                          # known models + install status
compiwer models download mtrini-svl-1.0       # resumable, verifies integrity
compiwer run Mtrini-SVL-1.0                   # interactive chat
compiwer serve --model Mtrini-SVL-1.0         # http://127.0.0.1:8000 (localhost only)
```

```python
from compiwer import Model, Agent, models, hardware

models.search("Mtrini")
print(hardware.info()["gpus"])

agent = Agent(model="Mtrini-SVL-1.0")          # safe read-only tools in v1
print(agent.run("List the Python files here and summarize them."))
```

## Local API (OpenAI-compatible)

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mtrini-svl-1.0",
       "messages": [{"role": "user", "content": "Hello!"}]}'
```

Use it from the official OpenAI client (`examples/openai_client.py`):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed-locally")
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (SSE with `stream:true`), `POST /v1/completions`, `GET /v1/hardware`. No key on localhost.

## Backends

| Backend | Format | Install | Notes |
|---|---|---|---|
| `llama_cpp` | GGUF | `pip install "compiwer[llama]"` | CPU + GPU, streaming, auto-detected |
| `transformers` | safetensors | `pip install "compiwer[transformers]"` | needs torch; adapters need their base |
| `image` | — | planned (diffusers) | clean interface today, honest error until then |

`Model("id", backend="auto")` picks from files on disk. Missing pieces produce actionable errors, never tracebacks-as-UX.

## Configuration

`~/.compiwer/config.toml` (+ `COMPIWER_HOME`, `COMPIWER_MODELS_DIR`, `COMPIWER_DEFAULT_BACKEND`, `COMPIWER_HOST`, `COMPIWER_PORT`, `HF_TOKEN`). Tokens are refused in config files — use env vars. See [docs](docs/).

## Mtrini Workspace integration

Workspace consumes this SDK as its model layer (no fake wiring — real APIs):

- `models.list/download/info` → model manager
- `runtime.process.start/stop/status` → runtime lifecycle
- `Model.chat/chat_stream` → inference
- `hardware.info()` → GPU/VRAM display
- `ImageBackend` + agent `ToolRegistry` → future image tools and agent tools

## Roadmap

Mtrini Imagine backend (diffusers) · GGUF metadata (quants/context) · model warm pools · `@compiwer/sdk` TypeScript client over the same REST API.

## Development / testing

```bash
pip install -e ".[all]"
python -m pytest -q            # unit tests (mocked inference, no weights needed)
python -m pytest -q -m integration  # needs network (tiny HF file)
```

License: Apache-2.0. Security reports: see SECURITY.md (private, please).
