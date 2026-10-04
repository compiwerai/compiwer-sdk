"""Streaming chat, token by token."""
from compiwer import Model

model = Model("mtrini-svl")
for chunk in model.chat_stream("Explain Python in one paragraph."):
    print(chunk.text, end="", flush=True)
print()
