# picogpt

**From Shakespeare-like dialogue to learning arithmetic: a character-level Transformer built in PyTorch.**

I wrote the model implementation by hand to understand attention, residual connections, normalization, and training. Then I turned it into an experiment: can next-token prediction learn addition rather than just memorize equations?

The latest recorded addition run answered **9,999 of 10,000 held-out equations correctly (99.99%)**, including **8,332/8,332 examples requiring a carry**. The remaining failure was `0+0`: the model generated `00000` instead of the expected `0`.

[Addition implementation](addition/main.py) · [Latest training log](addition/runs/20261017-174421-reversed-10k/training.log) · [Exact submitted code](addition/runs/20261017-174421-reversed-10k/submitted_main.py) · [Shakespeare sample](more.txt)

## The experiment

A small decoder-only Transformer learns character sequences such as `237+486=327\n`. The answer is reversed during training, so generation starts with the units digit. Predictions are reversed back for human-readable output.

The implementation includes:

- Learned token and position embeddings, causal multi-head attention, feedforward layers, residual connections, and pre-layer normalization.
- Synthetic addition data with operands from 0 to 999, sampled without duplicate ordered pairs.
- Train/validation splitting within all nine operand-length categories, preventing empty categories.
- Uniform category sampling during training, giving short equations attention despite their rarity in the full problem space.
- Answer-only cross-entropy: prompt and padding targets use `ignore_index=-100`.
- Greedy generation and exact-string answer evaluation, broken down by operand length and carry requirements.
- Local editing with Kaggle GPU submission, downloadable logs, source snapshots, and checkpoints.

## Recorded GPU results

| Run | Answer order | Batch | Context | Validation equations | Exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| [Normal-order baseline](addition/runs/20261017-164720-normal-order/training.log) | Most-significant first | 4 | 64 | 1,000 | 31.60% |
| [Reversed answers](addition/runs/20261017-172550-reversed-1k/training.log) | Units first | 64 | 16 | 1,000 | 100.00% |
| [Larger evaluation](addition/runs/20261017-174421-reversed-10k/training.log) | Units first | 64 | 16 | 10,000 | **99.99%** |

Each directory contains the actual submitted source and console log. These are separate runs, not a controlled ablation: batch size and context length changed alongside answer order, and initialization is not fixed. The improvement cannot be attributed to reversal alone.

Latest run configuration:

| Setting | Value |
| --- | --- |
| Selected equations | 300,000 |
| Training / validation equations | 270,000 / 30,000 |
| Embedding width / heads / blocks | 64 / 4 / 5 |
| Feedforward hidden width | 256 |
| Context length / batch size | 16 / 64 |
| Training steps | 10,000 |
| Optimizer / learning rate | AdamW / 0.001 |
| Dropout | 0 |
| Final train / validation loss | 0.0001 / 0.0403 |

The accuracy test samples 10,000 distinct validation equations with a fixed seed. Training and validation contain disjoint ordered pairs, but swapped equivalents such as `12+34` and `34+12` are not grouped together when splitting. This measures generalization to held-out ordered equations within the trained number range, not arbitrary-length arithmetic or proof of a particular internal algorithm. Validation loss uses balanced category sampling; the accuracy sample follows the validation dataset's natural proportions.

## What I learned

Early runs had decreasing loss but roughly 1% exact-answer accuracy. A diagnostic run memorized all 100 training equations, confirming that the model could fit a small dataset. That did not establish generalization.

The subsequent experiments focused on answer formatting, loss masking, category coverage, and exact-answer evaluation. Reversed output aligns generation with the direction of carry propagation, but the network is never given a hand-coded addition procedure. Python addition supplies labels and checks results; inference generates character IDs from learned weights.

## Run locally

```bash
git clone https://github.com/prashit-vora/picogpt.git
cd picogpt
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install torch numpy

# Addition: data is generated in the script.
cd addition
python3 main.py
```

Settings are at the top of `addition/main.py`. The script selects CUDA when available, otherwise CPU. Each invocation starts a fresh training run; there is no automatic checkpoint resume. After training it reports exact-answer accuracy and offers interactive testing.

## Run on Kaggle

```bash
pipx install kaggle
kaggle auth login
cd addition
python3 kaggle_run.py push
python3 kaggle_run.py status
python3 kaggle_run.py download
```

Before using another Kaggle account, change `KERNEL` in `addition/kaggle_run.py` to your own `username/slug`. Each push snapshots the latest saved `main.py` and submits a private GPU job. The runner checks CUDA availability, skips interactive input, and saves `training.log`, `submitted_main.py`, and `checkpoint.pt` under `/kaggle/working`. Downloads go into timestamped local directories. Checkpoints and generated job files are excluded from Git.

## Shakespeare

The original experiment remains at the repository root:

```bash
python3 main.py
```

It trains on `input.txt` and writes generated text to `more.txt`. The original reported run reached **1.4912 validation loss**, producing recognizable dialogue formatting and Shakespeare-like vocabulary, but inconsistent meaning. Its architecture uses 6 blocks, 6 heads, a 384-dimensional embedding, and a 256-character context. The root script does not save model weights.

## Repository layout

```text
main.py                 Shakespeare model and training
input.txt               Tiny Shakespeare corpus
more.txt                Generated Shakespeare-like text
addition/
  main.py               Addition training and evaluation
  kaggle_run.py         Remote job submission and output download
  runs/                 Recorded logs and exact submitted source
```

## Credits and reading

The starting architecture and learning path follow [Andrej Karpathy's GPT lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY). The corpus is [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare).

Relevant arithmetic reading: [Teaching Arithmetic to Small Transformers](https://arxiv.org/abs/2307.03381) and [Understanding Addition in Transformers](https://arxiv.org/abs/2310.13121). These informed the investigation; this repository is a learning project, not a reproduction of their full experimental setups.

## Next

Run a controlled comparison of answer order, investigate the zero-answer termination failure, then return to the ML systems roadmap: pretrained-model forward-pass verification, KV caching, and inference benchmarks.
