"""Compare pretrained GPT-2 and saved LoRA adapters on identical prompts."""

import argparse
import hashlib
import json
import math
from urllib.request import urlopen
from urllib.parse import urlencode
from pathlib import Path
import torch


def compare(model, adapters, idx, decode, max_new_tokens):
    scales = [layer.scale for layer in adapters]
    model.eval()
    with torch.no_grad():
        try:
            for layer in adapters:
                layer.scale = 0.0
            original = model.generate(
                idx.clone(),
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=True,
            )
        finally:
            for layer, scale in zip(adapters, scales):
                layer.scale = scale
        adapted = model.generate(
            idx.clone(), max_new_tokens=max_new_tokens, use_cache=True, do_sample=True
        )
    print("\nORIGINAL GPT-2:\n" + decode(original[0].tolist()))
    print("\nGPT-2 + LoRA:\n" + decode(adapted[0].tolist()))



def load_validation_tokens(main):
    """Cache the first 500 validation stories; never fetch training examples."""
    cache = Path(__file__).with_name("validation_stories.json")
    if cache.exists():
        stories = json.loads(cache.read_text())
    else:
        stories = []
        for offset in range(0, 500, 100):
            query = urlencode({"dataset": "roneneldan/TinyStories", "config": "default",
                               "split": "validation", "offset": offset, "length": 100})
            with urlopen("https://datasets-server.huggingface.co/rows?" + query,
                         timeout=60) as response:
                rows = json.load(response)["rows"]
            stories.extend(row["row"]["text"] for row in rows)
        if len(stories) != 500:
            raise ValueError("Expected 500 validation stories; cache was not saved.")
        cache.write_text(json.dumps(stories, ensure_ascii=False))
    if len(stories) != 500 or not all(isinstance(story, str) for story in stories):
        raise ValueError("Invalid validation-story cache.")
    return main.tokenize_stories(stories)


@torch.no_grad()
def evaluate_validation(main, adapters, checkpoint_path, windows=20, length=1024):
    data = load_validation_tokens(main)
    if len(data) <= length:
        raise ValueError("Validation data is shorter than a requested window.")
    # Build ONE collection of CPU windows and reuse it for both models.
    rng = torch.Generator().manual_seed(42)
    starts = torch.randint(len(data) - length, (windows,), generator=rng)
    batches = [(data[i:i + length].unsqueeze(0),
                data[i + 1:i + length + 1].unsqueeze(0)) for i in starts.tolist()]
    model = main.m
    scales = [layer.scale for layer in adapters]
    was_training = model.training
    model.eval()
    model.reset_cache()
    results = {}
    try:
        for label, enabled in (("pretrained", False), ("lora", True)):
            for layer, scale in zip(adapters, scales):
                layer.scale = scale if enabled else 0.0
            loss_sum = 0.0
            count = 0
            for number, (x, y) in enumerate(batches, 1):
                logits, loss = model(x.to(main.device), y.to(main.device), use_cache=False)
                loss_sum += loss.item() * y.numel()
                count += y.numel()
                del logits, loss
                print(f"{label}: {number}/{windows} windows", flush=True)
            mean_loss = loss_sum / count
            results[label] = {"loss": mean_loss, "perplexity": math.exp(mean_loss)}
    finally:
        for layer, scale in zip(adapters, scales):
            layer.scale = scale
        model.reset_cache()
        model.train(was_training)
    report = {"dataset": "roneneldan/TinyStories", "split": "validation",
              "stories": 500, "seed": 42, "window_length": length,
              "window_starts": starts.tolist(), "tokens_per_model": windows * length,
              "token_data_sha256": hashlib.sha256(data.numpy().tobytes()).hexdigest(),
              "checkpoint": str(checkpoint_path.resolve()), "results": results}
    output = checkpoint_path.with_name("validation_comparison.json")
    output.write_text(json.dumps(report, indent=2))
    for label, values in results.items():
        print(f"{label}: loss={values['loss']:.4f}, perplexity={values['perplexity']:.2f}")
    print("Saved", output)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(__file__).with_name("lora_adapters.pt")
    )
    parser.add_argument("--prompt", help="One prompt; otherwise ask repeatedly.")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--evaluate", action="store_true",
                        help="Compare loss on fixed validation windows, then exit.")
    parser.add_argument("--eval-windows", type=int, default=20)
    parser.add_argument("--eval-length", type=int, default=1024)
    args = parser.parse_args()
    if args.eval_windows < 1 or not 1 <= args.eval_length <= 1024:
        parser.error("Use positive --eval-windows and --eval-length between 1 and 1024.")
    if not args.checkpoint.is_file():
        parser.error(
            f"Adapter checkpoint not found: {args.checkpoint}. Train first or pass --checkpoint PATH."
        )
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not checkpoint.get("training_completed", False):
        parser.error("Checkpoint does not report completed training.")
    config = checkpoint["config"]
    if config.get("rank") != 8:
        parser.error("This model currently uses rank 8 adapters.")
    import main  # Loads the base model and creates adapters; does not train/download data.

    expected = {
        name: p
        for name, p in main.m.named_parameters()
        if name.endswith((".A.weight", ".B.weight"))
    }
    saved = checkpoint["adapters"]
    if saved.keys() != expected.keys():
        raise ValueError("Adapter checkpoint keys do not match this model.")
    with torch.no_grad():
        for name, parameter in expected.items():
            if saved[name].shape != parameter.shape:
                raise ValueError(f"Wrong adapter shape: {name}")
            parameter.copy_(saved[name])
    adapters = [
        layer for layer in main.m.modules() if isinstance(layer, main.LoRALinear)
    ]
    for layer in adapters:
        layer.scale = config["alpha"] / config["rank"]
    if args.evaluate:
        evaluate_validation(main, adapters, args.checkpoint, args.eval_windows, args.eval_length)
        return
    print("Loaded trained adapters. Empty input exits.")
    while True:
        try:
            prompt = args.prompt if args.prompt is not None else input("\nPrompt: ")
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt.strip():
            break
        idx = torch.tensor([main.encode(prompt)], dtype=torch.long, device=main.device)
        compare(main.m, adapters, idx, main.decode, args.max_new_tokens)
        if args.prompt is not None:
            break


if __name__ == "__main__":
    run()
