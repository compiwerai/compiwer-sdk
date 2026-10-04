"""Serve a model locally, then call it like OpenAI.

Terminal 1:  python examples/local_server.py
Terminal 2:  curl http://127.0.0.1:8000/v1/models
"""
from compiwer.runtime.server import serve

serve("mtrini-svl-1.0", host="127.0.0.1", port=8000)
