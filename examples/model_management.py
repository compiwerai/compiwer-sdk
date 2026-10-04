"""Model management: list, search, download, info, delete."""
from compiwer import models

print("Known:", [m["id"] for m in models.list()])
print("Search:", [m["id"] for m in models.search("Mtrini")][:5])

models.download("CompiwerAI/Mtrini-SVL-1.0-GGUF")
print("Exists:", models.exists("mtrini-svl-1.0"))
print(models.info("mtrini-svl-1.0"))
