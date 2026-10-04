"""Local runtime: foreground server with a friendly banner. Localhost by default."""
from __future__ import annotations

from typing import Any


def _require_server() -> None:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ModuleNotFoundError:
        from ..exceptions import RuntimeError

        raise RuntimeError('Serving needs FastAPI + Uvicorn.',
                           fix='pip install "compiwer[server]"') from None


def serve(model_id: str = "mtrini-svl-1.0", host: str = "127.0.0.1", port: int = 8000,
          backend: str = "auto", public: bool = False, **kwargs: Any) -> None:
    _require_server()
    import uvicorn

    from ..api.app import create_app
    from ..client import Model
    from ..utils import hardware as _hw

    if host not in ("127.0.0.1", "localhost", "::1") and not public:
        from ..exceptions import RuntimeError as _RE

        raise _RE(f"Refusing to bind {host} without --public.",
                  cause="Compiwer never exposes your runtime to the network automatically.",
                  fix="Use --host 127.0.0.1, or pass --public if you truly want LAN access.")
    model = Model(model_id, backend=backend)
    model._ensure()  # warm now so errors surface before serving (raises helpfully)
    try:
        from rich.console import Console
        from rich.panel import Panel
    except ModuleNotFoundError:
        print(f"Compiwer Runtime — {model_id} on http://{host}:{port}")
    else:
        hw = _hw.info()
        dev = ", ".join(hw["gpus"]) if hw["gpus"] else "CPU"
        Console().print(Panel(
            f"Model:   {model_id}\nBackend: {model.backend_name}\nDevice:  {dev}\n"
            f"Address: http://{host}:{port}\nStatus:  Ready",
            title="Compiwer Runtime", expand=False))
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
