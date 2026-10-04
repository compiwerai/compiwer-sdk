"""Use the local Compiwer server through the official OpenAI client.

Terminal 1:  compiwer serve --model mtrini-svl-1.0
Terminal 2:  python examples/openai_client.py
"""
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed-locally")

resp = client.chat.completions.create(
    model="mtrini-svl-1.0",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
