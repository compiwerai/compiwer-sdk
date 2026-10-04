# Contributing to Compiwer SDK

## Setup

```bash
pip install -e ".[all]"
python -m pytest -q
```

## Rules

- Python 3.11+, typed, `ruff` clean.
- Lazy imports for heavy deps (torch/transformers/llama_cpp/fastapi) — the SDK imports in a bare env.
- Every backend/provider path gets a test with mocks; no weights in CI.
- Errors must say what failed + how to fix it (see `exceptions.py`).
- Never commit secrets, weights, or `~/.compiwer` contents.
- Docs live with code: update `docs/` + README in the same change.

## Pull requests

Describe what you ran (tests, `compiwer doctor`, manual checks). Security issues go privately per SECURITY.md.
