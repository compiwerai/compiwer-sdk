"""Registry: ids, aliases, ad-hoc repos, unknown ids."""
import pytest

from compiwer.exceptions import ModelNotFoundError
from compiwer.models.registry import all_models, resolve


def test_aliases():
    assert resolve("mtrini-svl").id == "mtrini-svl-1.0"
    assert resolve("MTRINI-SVL-1.0").id == "mtrini-svl-1.0"
    assert resolve("mtrini-tellus").id == "mtrini-tellus-27b"


def test_raw_repo():
    m = resolve("CompiwerAI/Mtrini-SVL-1.0-GGUF")
    assert m.backend == "llama_cpp"
    m2 = resolve("Qwen/Qwen2.5-0.5B-Instruct")
    assert m2.backend == "transformers"


def test_unknown():
    with pytest.raises(ModelNotFoundError) as e:
        resolve("nope-not-a-model")
    assert "mtrini-svl-1.0" in str(e.value)


def test_all_have_backends():
    assert all(m.backend in ("llama_cpp", "transformers", "image") for m in all_models())
