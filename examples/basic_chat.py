"""Basic chat: download once, chat forever (locally)."""
from compiwer import Model

model = Model("Mtrini-SVL-1.0")  # backend auto-detected

print(model.chat("Write a Python calculator.").text)

print(model.chat([
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Build a calculator."},
]).text)
