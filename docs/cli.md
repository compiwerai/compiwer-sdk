# CLI

```
compiwer models list|search|download|info|delete
compiwer run [MODEL] [--backend auto|llama_cpp|transformers]
compiwer serve [--model …] [--host 127.0.0.1] [--port 8000] [--backend …] [--public]
compiwer hardware
compiwer doctor
compiwer version
```

Global flags: `--verbose` (debug), `--quiet` (errors only).
`models delete` asks for confirmation unless `--yes`.
Secrets via env (`HF_TOKEN`); `--token` is accepted but prefer env so shells don't log it.
