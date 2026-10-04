# Local API

`compiwer serve --model mtrini-svl-1.0` → `http://127.0.0.1:8000` (localhost only;
`--host` + `--public` to expose deliberately).

- `GET /health`, `GET /v1/models`, `GET /v1/hardware`
- `POST /v1/chat/completions` (`stream: true` → SSE + `[DONE]`)
- `POST /v1/completions`

Works with the official OpenAI client via `base_url="http://127.0.0.1:8000/v1"`.
No key on localhost. The schema is the stable surface a future `@compiwer/sdk`
TypeScript package will target.
