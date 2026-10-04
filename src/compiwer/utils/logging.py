"""Logging: quiet by default, --verbose for debug, secrets always redacted."""
from __future__ import annotations

import logging
import os
import re

_SECRET_RE = re.compile(r"(hf_[A-Za-z0-9_-]{8,}|sk-[A-Za-z0-9-]{8,}|api[_-]?key\s*[:=]\s*\S+)", re.IGNORECASE)


class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _SECRET_RE.sub("***", str(record.getMessage()))
        record.args = ()
        return True


def setup(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else (logging.ERROR if quiet else logging.WARNING)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    log = logging.getLogger("compiwer")
    log.handlers = [handler]
    log.setLevel(level)
    log.addFilter(_RedactingFilter())
    log.propagate = False
    return log


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None
