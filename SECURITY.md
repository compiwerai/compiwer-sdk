# Security Policy

## Reporting

Do **not** open public issues for vulnerabilities. Contact the Compiwer AI maintainers privately with version, reproduction, and impact.

## Local-first guarantees

- The runtime binds `127.0.0.1` by default; LAN bind needs explicit `--public`.
- No shell execution in v1 agent tools; dangerous tools are opt-in (`allow_dangerous`).
- No telemetry. No uploads to Compiwer. Prompts stay local unless you configure a remote endpoint.
- Tokens live in env vars, never in config files (refused), logs (redacted), or errors.
- Downloads verify integrity; interrupted downloads resume; nothing is auto-deleted.
