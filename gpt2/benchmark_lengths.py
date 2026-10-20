"""Measure cached/uncached generation across exact input-token lengths.

Defaults match the initial single-run length experiment. Increase --trials
for repeated measurements; loading, tokenization, and printing are not timed.
"""
import argparse
import time
import torch
import gpt2 as main

parser = argparse.ArgumentParser()
parser.add_argument("--lengths", type=int, nargs="+", default=[16, 64, 256, 512])
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--new-tokens", type=int, default=100)
args = parser.parse_args()
if args.trials < 1 or args.new_tokens < 1 or any(n < 1 for n in args.lengths):
    parser.error("Lengths, trials, and new-tokens must be positive")
if max(args.lengths) + args.new_tokens > main.block_size:
    parser.error("Keep input + output within the context window for this experiment")

main.m.eval()
all_ids = main.encode("The cat sat on the mat. " * 2000)
assert len(all_ids) >= max(args.lengths)

def synchronize():
    if main.device.type == "cuda":
        torch.cuda.synchronize()

# Untimed warm-up for both modes, using the shortest requested input.
warmup = torch.tensor([all_ids[:min(args.lengths)]], dtype=torch.long, device=main.device)
for use_cache in (False, True):
    main.m.generate(warmup, 5, use_cache=use_cache, do_sample=False)
synchronize()

print("Input tokens | Uncached seconds | Cached seconds | Speedup", flush=True)
for length in args.lengths:
    idx = torch.tensor([all_ids[:length]], dtype=torch.long, device=main.device)
    totals = {False: 0.0, True: 0.0}
    for trial in range(args.trials):
        outputs = {}
        for use_cache in (False, True):
            synchronize()
            start = time.perf_counter()
            outputs[use_cache] = main.m.generate(
                idx, args.new_tokens, use_cache=use_cache, do_sample=False
            )
            synchronize()
            elapsed = time.perf_counter() - start
            totals[use_cache] += elapsed
            print(f"length={length} trial={trial + 1} cache={use_cache}: {elapsed:.6f}s", flush=True)
        if not torch.equal(outputs[False], outputs[True]):
            print("WARNING: generated token IDs differ for this pair", flush=True)
    uncached, cached = totals[False] / args.trials, totals[True] / args.trials
    print(f"{length:12d} | {uncached:16.6f} | {cached:14.6f} | {uncached/cached:.2f}x", flush=True)
