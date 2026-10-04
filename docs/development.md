# Development

```bash
pip install -e ".[all]"
python -m pytest -q
python -m pytest -q -m integration   # network: one tiny HF file
ruff check src tests 2>/dev/null || true
```

Conventions: lazy heavy imports, mocked inference in tests, friendly errors,
docs updated with code. See CONTRIBUTING.md.
