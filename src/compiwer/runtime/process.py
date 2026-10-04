"""Background runtime management (pidfile): start/stop/status for Workspace integration."""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

from ..utils.paths import log_file, pid_file


def _read_pid() -> int | None:
    p = pid_file()
    if not p.exists():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            import ctypes

            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not h:
                return False
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def status() -> dict[str, Any]:
    pid = _read_pid()
    if pid and _alive(pid):
        return {"running": True, "pid": pid}
    return {"running": False, "pid": None}


def start(model_id: str = "mtrini-svl-1.0", host: str = "127.0.0.1", port: int = 8000,
          backend: str = "auto") -> dict[str, Any]:
    st = status()
    if st["running"]:
        return {"running": True, "pid": st["pid"], "note": "already running"}
    log = open(log_file(), "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "compiwer.cli.main", "serve", "--model", model_id,
         "--host", host, "--port", str(port), "--backend", backend],
        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
        creationflags=(0x08000000 if os.name == "nt" else 0))  # CREATE_NO_WINDOW
    pid_file().write_text(str(proc.pid), encoding="utf-8")
    return {"running": True, "pid": proc.pid, "address": f"http://{host}:{port}"}


def stop() -> dict[str, Any]:
    pid = _read_pid()
    if not pid:
        return {"running": False, "note": "no runtime recorded"}
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=15)
        else:
            os.kill(pid, 15)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        pid_file().unlink()
    except OSError:
        pass
    return {"running": False, "note": f"stopped pid {pid}"}
