# GPT-2 LoRA on TinyStories

QKV-only LoRA on the handwritten GPT-2 small implementation. Original weights are frozen; each attention head gets rank-8 adapters, alpha 8, with a randomly initialized A and zero-initialized B. There are **1,105,920 trainable parameters**, out of 125,545,728 including adapters.

## Recorded experiment

- Data: 5,000 stories from the shuffled TinyStories training stream (seed 42, buffer 10,000); first 500 validation stories kept separate.
- GPT-2 tokenizer, end-of-text separators, contiguous 1,024-token windows, next-token loss on all positions. Windows may cross story boundaries.
- One Tesla T4, float32, batch 4, AdamW learning rate 1e-4, 10,000 updates, no dropout.
- Training loss 2.3722 → 1.7833; sampled validation loss 2.2782 → 1.7372.
- Measured run time including data preparation, evaluation and final samples: **17,615.85 seconds (~4 h 54 min)**. Peak allocated CUDA memory **7.99 GiB**.
- Earlier batch-24 attempt ran out of GPU memory during initial evaluation.

## Fixed validation comparison

Both versions saw the identical 20 windows of 1,024 tokens (20,480 target tokens), selected with seed 42 from the tokenized validation stories. Batch size one, evaluation mode, cache disabled, token-weighted cross-entropy. The baseline disables adapter corrections, leaving the frozen pretrained weights. No temperature or sampling is used for this loss measurement.

| Model | Cross-entropy loss | Perplexity |
| --- | ---: | ---: |
| Pretrained GPT-2 | 2.2679 | 9.66 |
| GPT-2 + LoRA | 1.7184 | 5.58 |

Perplexity is exp(loss), not accuracy. Window starts and the validation token hash are retained in [the evaluation report](results/2026-10-22/validation_comparison.json). This is a sampled comparison, not an exhaustive benchmark; sampled windows can overlap. Evaluation supplies ground-truth preceding tokens, unlike free-running generation.

## Generation results and limitations

The [original run log](results/2026-10-22/training.log) contains paired greedy outputs for two hand-selected prompts (not reference story openings), limited to 80 new tokens. GPT-2 repeated “She was a little bit of a rabbit”; LoRA introduced Timmy, a ball and his mother. The adapted outputs still contain inconsistencies.

Later interactive tests used temperature 0.8, top-k 40 and EOS stopping. Some outputs avoided exact loops, while others still repeated, contradicted earlier sentences or confused characters. The demonstrated result is improved held-out token prediction and adaptation to story style, not reliably coherent storytelling. These later sampled outputs are not the greedy outputs in the recorded training log.

## Run locally

From the repository root, preferably in a fresh Python environment:

```bash
python3 -m pip install -r lora/requirements.txt
python3 lora/download_model.py
python3 lora/prompt.py --checkpoint lora/results/2026-10-22/lora_adapters.pt
python3 lora/prompt.py --checkpoint lora/results/2026-10-22/lora_adapters.pt --evaluate
```

The small trained adapter checkpoint is included. The pretrained GPT-2 checkpoint is downloaded separately. Validation mode caches the first 500 validation stories locally and evaluates both models on the same windows. It saves validation_comparison.json beside the selected checkpoint; rerunning overwrites that report. Interactive generation samples, so outputs vary.

Training: edit settings in main.py and run `python3 lora/main.py`. Importing main does not fetch training data or train. Training saves lora_adapters.pt beside main.py.

## Kaggle terminal workflow

Authenticate the Kaggle CLI, install it separately if needed, and change `KERNEL` in kaggle_run.py to a private notebook ID under your own account before submitting.

```bash
python3 lora/kaggle_run.py prepare
python3 lora/kaggle_run.py push
python3 lora/kaggle_run.py status
python3 lora/kaggle_run.py download
```

The launcher uses the saved main.py training settings without overriding batch size or context length. It adapts asset paths, downloads the pinned GPT-2 checkpoint on Kaggle, and saves logs/adapters plus sequential baseline/LoRA samples. Internet and GPU access are required. The exact submitted training source and configuration are retained under results/2026-10-22. The recorded run predates the later sampling/EOS changes.

Sources: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and [GPT-2](https://huggingface.co/openai-community/gpt2). Respect their respective data/model licenses. Dependency versions and the TinyStories revision were not pinned for the recorded run; the fixed-evaluation token hash helps identify data differences.
