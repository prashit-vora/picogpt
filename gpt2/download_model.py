"""Download the original GPT-2 small checkpoint, separately from this repository."""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

model_dir = Path(os.environ.get("GPT2_MODEL_DIR", Path(__file__).resolve().parent / "model"))
snapshot_download(
    repo_id="openai-community/gpt2",
    revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
    local_dir=str(model_dir),
    allow_patterns=["config.json", "generation_config.json", "model.safetensors",
                    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                    "README.md"],
)
print(f"GPT-2 downloaded to {model_dir}")
