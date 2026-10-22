"""Download GPT-2 base assets beside main.py; weights are not stored in Git."""
from pathlib import Path
from huggingface_hub import snapshot_download
snapshot_download("openai-community/gpt2",
    revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
    local_dir=str(Path(__file__).resolve().parent),
    allow_patterns=["model.safetensors", "tokenizer.json"])
