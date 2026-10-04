# Backends

`Backend`: `load / unload / chat / chat_stream / generate / capabilities`.
Auto-detection reads the model folder (`.gguf` → `llama_cpp`, safetensors → `transformers`).

- **llama_cpp**: `pip install "compiwer[llama]"`. Picks a mid-size quant when several
  are bundled (`filename=` overrides). GPU layers on by default when CUDA is present.
  For custom CUDA builds see llama-cpp-python docs (set `CMAKE_ARGS` before install).
- **transformers**: `pip install "compiwer[transformers]"` (+ torch; CPU users see
  pytorch.org for the CPU wheel). `device_map="auto"` on CUDA, float16 there,
  float32 on CPU. Streaming via `TextIteratorStreamer`.
- **image**: interface only in v1 (`generate/edit`); `ImageModel` raises an honest
  `BackendUnavailableError` until a diffusers backend registers.

Add a backend by subclassing `Backend` and registering in `backends/__init__.py`.
