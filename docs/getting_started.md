# Getting started

```bash
pip install "compiwer[all]"
compiwer doctor
compiwer models download mtrini-svl-1.0
compiwer run Mtrini-SVL-1.0
```

```python
from compiwer import Model
print(Model("mtrini-svl").chat("Hello!").text)
```

Models live in `~/.compiwer/models/` (override: `COMPIWER_HOME` or `COMPIWER_MODELS_DIR`).
Private/gated repos need `HF_TOKEN` in the environment.
