"""Configuration: ~/.compiwer/config.toml + environment overrides. Secrets are never printed."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError
from .utils.paths import config_file


def _parse_toml(text: str) -> dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        raise ConfigurationError("TOML support needs Python 3.11+.",
                                 fix="Upgrade Python, or use environment variables instead.")
    try:
        data = tomllib.loads(text)
    except ValueError as e:
        raise ConfigurationError(f"Invalid config file: {e}", fix="Fix or delete the config file.")
    return data if isinstance(data, dict) else {}


def load() -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    p = config_file()
    if p.exists():
        try:
            cfg = _parse_toml(p.read_text(encoding="utf-8"))
        except OSError as e:
            raise ConfigurationError(f"Cannot read config file: {e}")
    env_map = {
        "COMPIWER_HOME": ("home",),
        "COMPIWER_MODELS_DIR": ("models_dir",),
        "COMPIWER_DEFAULT_BACKEND": ("default_backend",),
        "COMPIWER_HOST": ("host",),
        "COMPIWER_PORT": ("port",),
    }
    for var, _key in env_map.items():
        if os.environ.get(var):
            cfg[_key[0]] = os.environ[var]
    return cfg


def get(key: str, default: Any = None) -> Any:
    return load().get(key, default)


def save(patch: dict[str, Any]) -> Path:
    """Merge patch into config.toml (flat keys only). Never stores tokens."""
    p = config_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    current: dict[str, Any] = {}
    if p.exists():
        current = _parse_toml(p.read_text(encoding="utf-8"))
    for k, v in patch.items():
        if "token" in k.lower() or "key" in k.lower():
            raise ConfigurationError(f"Refusing to store secret-like key {k!r} in config.",
                                     fix="Export it as an environment variable instead (e.g. HF_TOKEN).")
        current[k] = v
    lines = [f"{k} = {json_value(v)}\n" for k, v in sorted(current.items())]
    p.write_text("".join(lines), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return p


def json_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return '"' + str(v).replace('"', "'") + '"'


def masked(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    for k in list(out):
        if "token" in k.lower() or "key" in k.lower():
            out[k] = "***"
    out["HF_TOKEN"] = "***" if os.environ.get("HF_TOKEN") else None
    return out
