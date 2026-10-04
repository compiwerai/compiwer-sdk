"""Compiwer CLI — models, runtime, hardware, doctor. Rich output, honest errors."""
from __future__ import annotations

import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # Windows consoles/pipes default to cp1252; never crash on ✓
    except Exception:
        pass

try:
    import typer
    from rich.console import Console
    from rich.table import Table
except ModuleNotFoundError:  # pragma: no cover
    raise ImportError("The CLI needs typer + rich: pip install compiwer")

from .. import __version__
from ..exceptions import CompiwerError
from ..models import manager as models
from ..utils import hardware as _hw
from ..utils import logging as _log

app = typer.Typer(help="Compiwer SDK — Build AI locally.", no_args_is_help=True)
console = Console()
models_app = typer.Typer(help="Download and manage local models.", no_args_is_help=True)
app.add_typer(models_app, name="models")


def _err(e: CompiwerError) -> None:
    console.print(f"[red]Error:[/red] {e}")
    raise typer.Exit(1)


@app.callback()
def _global(verbose: bool = typer.Option(False, "--verbose", help="Debug logs."),
            quiet: bool = typer.Option(False, "--quiet", help="Errors only.")) -> None:
    _log.setup(verbose=verbose, quiet=quiet)


@app.command()
def version() -> None:
    """Print the SDK version."""
    console.print(f"compiwer {__version__}")


@models_app.command("list")
def models_list() -> None:
    """List known models and install status."""
    t = Table("id", "backend", "installed", "size")
    for m in models.list():
        t.add_row(m["id"], m["backend"], "yes" if m["installed"] else "no", m["size"] or "—")
    console.print(t)


@models_app.command("search")
def models_search(query: str, limit: int = 20) -> None:
    """Search Hugging Face for models."""
    try:
        for m in models.search(query, limit):
            console.print(f"[bold]{m.get('id')}[/bold]  ♥{m.get('likes', 0)}")
    except CompiwerError as e:
        _err(e)


@models_app.command("download")
def models_download(model_id: str,
                    file: str | None = typer.Option(None, "--file", help="Single filename instead of snapshot."),
                    force: bool = typer.Option(False, "--force", help="Re-download."),
                    token: str | None = typer.Option(None, "--token", help="HF token (or HF_TOKEN).")) -> None:
    """Download a model (resumable)."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    try:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as bar:
            bar.add_task(f"Downloading {model_id}…", total=None)
            dest = models.download(model_id, filename=file, force=force)
        from ..models.metadata import dir_size, fmt_size

        console.print(f"[green]✓[/green] {model_id} → {dest} ({fmt_size(dir_size(dest))})")
    except CompiwerError as e:
        _err(e)


@models_app.command("info")
def models_info(model_id: str) -> None:
    """Show registry + local + Hub details for a model."""
    import json

    try:
        console.print_json(json.dumps(models.info(model_id), indent=1, default=str))
    except CompiwerError as e:
        _err(e)


@models_app.command("delete")
def models_delete(model_id: str, yes: bool = typer.Option(False, "--yes", help="Skip confirmation.")) -> None:
    """Delete a downloaded model (asks first)."""
    try:
        if not yes and not typer.confirm(f"Delete {model_id} from disk?"):
            raise typer.Abort()
        models.delete(model_id)
        console.print(f"[green]✓[/green] Deleted {model_id}")
    except CompiwerError as e:
        _err(e)


@app.command()
def run(model_id: str = typer.Argument("mtrini-svl-1.0"),
        backend: str = typer.Option("auto", "--backend", help="llama_cpp | transformers | auto")) -> None:
    """Interactive local chat (Ctrl+C to quit)."""
    from ..client import Model

    try:
        model = Model(model_id, backend=backend)
        console.print(f"[bold]Chatting with {model_id}[/bold] (empty line quits)")
        while True:
            try:
                text = console.input("[cyan]you>[/cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                break
            try:
                console.print(f"[green]mtrini>[/green] {model.chat(text).text}\n")
            except CompiwerError as e:
                console.print(f"[red]Error:[/red] {e}")
                break
    except CompiwerError as e:
        _err(e)


@app.command()
def serve(model: str = typer.Option("mtrini-svl-1.0", "--model"),
          host: str = typer.Option("127.0.0.1", "--host"),
          port: int = typer.Option(8000, "--port"),
          backend: str = typer.Option("auto", "--backend"),
          public: bool = typer.Option(False, "--public", help="Allow non-localhost bind.")) -> None:
    """Start the local OpenAI-compatible server."""
    from ..runtime import server as _srv

    try:
        _srv.serve(model, host=host, port=port, backend=backend, public=public)
    except CompiwerError as e:
        _err(e)


@app.command()
def hardware() -> None:
    """Show CPU/RAM/GPU/CUDA/disk report."""
    t = Table("component", "detail")
    for label, detail, _ok in _hw.describe():
        t.add_row(label, detail)
    console.print(t)


@app.command()
def doctor() -> None:
    """Diagnose the machine with actionable fixes."""
    console.print("[bold]Compiwer Doctor[/bold]\n")
    ok_all = True

    def row(label: str, ok: bool, hint: str = "") -> None:
        nonlocal ok_all
        ok_all = ok_all and ok
        console.print(f"[green]✓[/green] {label}" if ok else f"[red]✗[/red] {label}")
        if hint and not ok:
            console.print(f"  → {hint}")

    import platform

    row(f"Python {platform.python_version()}", True)
    row("Compiwer SDK", True, "")
    try:
        import huggingface_hub  # noqa: F401

        row("Hugging Face Hub", True)
    except ModuleNotFoundError:
        row("Hugging Face Hub", False, 'pip install "compiwer[all]"')
    try:
        import httpx

        r = httpx.get("https://huggingface.co/api/models?limit=1", timeout=10)
        row("Hugging Face connectivity", r.status_code == 200, "Check your connection (public models need no token).")
    except Exception:
        row("Hugging Face connectivity", False, "Check your connection.")
    info = _hw.info()
    row(f"GPU ({', '.join(info['gpus']) if info['gpus'] else 'CPU mode'})", True)
    row("CUDA", info["cuda"], "CPU inference still works; install CUDA torch for GPU.")
    for name in ("llama_cpp", "transformers"):
        try:
            from ..backends import get as _get

            avail, reason = _get(name).is_available()
        except Exception as e:  # pragma: no cover
            avail, reason = False, str(e)
        row(f"Backend {name}", avail, reason)
    row(f"Disk ({info['disk_free_gb']} GB free)", (info["disk_free_gb"] or 0) > 2,
        "Free space or set COMPIWER_MODELS_DIR elsewhere.")
    console.print("\n[bold]Models[/bold]")
    any_installed = False
    for m in models.list():
        if m["installed"]:
            any_installed = True
            console.print(f"[green]✓[/green] {m['id']} ({m['size']})")
    if not any_installed:
        console.print("  (none) → compiwer models download mtrini-svl-1.0")
    console.print("\n" + ("[green]System ready.[/green]" if ok_all and any_installed else "[yellow]Fix the ✗ items above, then retry.[/yellow]"))
