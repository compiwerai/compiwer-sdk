"""Model manager + downloader paths (network-free except skipped integration)."""
import pytest

from compiwer import models
from compiwer.exceptions import ModelNotFoundError


def test_list_and_exists(home, gguf_dir):
    rows = models.list()
    svl = next(r for r in rows if r["id"] == "mtrini-svl-1.0")
    assert svl["installed"] is True and svl["size"] is not None
    assert models.exists("mtrini-svl") is True
    assert models.exists("mtrini-tellus-27b") is False


def test_info_local(home, gguf_dir):
    info = models.info("mtrini-svl-1.0")
    assert info["format"] == "GGUF" and info["installed"] is True


def test_delete_missing(home):
    with pytest.raises(ModelNotFoundError):
        models.delete("mtrini-svl-1.0")


def test_delete_roundtrip(home, gguf_dir):
    assert models.delete("mtrini-svl-1.0") is True
    assert models.exists("mtrini-svl-1.0") is False


@pytest.mark.integration()
def test_download_tiny_gguf(home):
    """Needs network + HF. Runs only with -m integration."""
    from compiwer.models import downloader

    p = downloader.download("hf-internal-testing/tiny-random-gpt2", filename="config.json")
    assert (p / "config.json").exists()
