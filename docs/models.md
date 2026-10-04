# Models

Curated registry (`compiwer.models.registry`): `mtrini-svl-1.0` (+`mtrini-svl`),
`mtrini-svl-1.1`, `mtrini-tellus-27b` (+`mtrini-tellus`), `mtrini-tellus-12b-sahara-2`,
`mtrini-imagine-1.0-7b` (image, planned backend).

```python
from compiwer import models
models.list()                                   # installed? size? backend?
models.search("Mtrini")
models.download("CompiwerAI/Mtrini-SVL-1.0-GGUF")  # or short id, or --file one .gguf
models.info("mtrini-svl-1.0")                   # registry + local + Hub
models.delete("mtrini-svl-1.0")                 # asks first in CLI, explicit in API
models.exists("mtrini-svl-1.0")
```

Any `Org/Repo` id also works directly (GGUF→llama_cpp, else transformers).
Snapshots resume; sizes are checked against free disk before fetching.
