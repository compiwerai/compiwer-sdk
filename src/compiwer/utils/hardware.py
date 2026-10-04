"""Best-effort hardware detection (stdlib only — works on CPU-only machines)."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any


def _mem_bytes() -> int | None:
    try:
        if os.name == "nt":
            import ctypes

            class Stat(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            s = Stat()
            s.dwLength = ctypes.sizeof(Stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
                return int(s.ullTotalPhys)
        elif os.path.exists("/proc/meminfo"):
            for line in open("/proc/meminfo", encoding="utf-8"):
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        else:  # macOS
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=10)
            if out.returncode == 0:
                return int(out.stdout.strip())
    except Exception:
        pass
    return None


def _nvidia() -> tuple[list[str], int | None]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return [], None
        names, vram = [], 0
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            names.append(parts[0])
            if len(parts) > 1 and "MiB" in parts[1]:
                vram += int(parts[1].split()[0])
        return names, (vram * 1024 * 1024 if vram else None)
    except (OSError, ValueError):
        return [], None


def _cuda() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        gpus, _ = _nvidia()
        return bool(gpus)


def _disk_gb(path: str) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return -1.0


def info() -> dict[str, Any]:
    from .paths import models_dir

    gpus, vram = _nvidia()
    mem = _mem_bytes()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine() or "unknown",
        "python": platform.python_version(),
        "ram_gb": round(mem / 1e9, 1) if mem else None,
        "gpus": gpus,
        "vram_gb": round(vram / 1e9, 1) if vram else None,
        "cuda": _cuda(),
        "disk_free_gb": round(_disk_gb(str(models_dir())), 1),
        "models_dir": str(models_dir()),
    }


def describe() -> list[tuple[str, str, bool]]:
    """Rows for the CLI: (label, detail, ok)."""
    i = info()
    return [
        ("OS", i["os"], True),
        ("CPU", i["cpu"], True),
        ("Python", i["python"], True),
        ("RAM", f"{i['ram_gb']} GB" if i["ram_gb"] else "unknown", i["ram_gb"] is not None),
        ("GPU", ", ".join(i["gpus"]) if i["gpus"] else "none (CPU mode)", True),
        ("VRAM", f"{i['vram_gb']} GB" if i["vram_gb"] else ("n/a" if not i["gpus"] else "unknown"), True),
        ("CUDA", "available" if i["cuda"] else "not available (CPU inference still works)", True),
        ("Disk free", f"{i['disk_free_gb']} GB at {i['models_dir']}", (i["disk_free_gb"] or 0) > 1),
    ]
