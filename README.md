# picogpt

**From training a character-level Transformer to running pretrained GPT-2 with a custom forward pass and KV caching.**

I wrote the model implementation by hand to understand attention, residual connections, normalization, and training. Then I turned it into an experiment: can next-token prediction learn addition rather than just memorize equations?

The latest recorded addition run answered **9,999 of 10,000 held-out equations correctly (99.99%)**, including **8,332/8,332 examples requiring a carry**. The remaining failure was `0+0`: the model generated `00000` instead of the expected `0`.

[Addition implementation](addition/main.py) · [Latest training log](addition/runs/20261017-174421-reversed-10k/training.log) · [Exact submitted code](addition/runs/20261017-174421-reversed-10k/submitted_main.py) · [Shakespeare sample](more.txt)

## GPT-2 inference: from weights to cached decoding

The next stage is a handwritten PyTorch forward pass for **GPT-2 small (124M)**, loading the original pretrained weights rather than retraining the model. The earlier character-level models remain below; GPT-2 uses its original byte-level BPE tokenizer.

- **12 blocks, 12 attention heads, width 768, context 1,024**, tanh-approximate GELU, and tied token-embedding/output weights.
- Explicit checkpoint mapping: transpose GPT-2 projection weights and split combined QKV parameters across separate attention heads.
- **KV caching per head and layer**: prefill the prompt, then process only the newest token, with offset position embeddings and causal masks.
- Cache reset between prompts and rebuilding when the 1,024-token window slides; ordinary uncached forward remains available.
- Only the last position is projected to vocabulary scores during generation.

### Correctness checks

The packaged CPU comparison against Hugging Face GPT-2 on `The capital of France is` had a **maximum logit difference of 4.58e-5** and identical greedy-generated token IDs (cached custom model versus reference). Cache checks on the pretrained model had a maximum difference around **6.87e-5**, with identical cached/uncached greedy continuations. These are checks on selected inputs, not an exhaustive guarantee or a quality benchmark.

`prompt.py` compares logits with Hugging Face and displays cached custom-model output alongside reference greedy output.

### Run GPT-2

From the repository root, in your Python environment:

```bash
python3 -m pip install -r gpt2/requirements.txt
python3 gpt2/download_model.py
python3 gpt2/prompt.py
```

The checkpoint is downloaded at a pinned revision into ignored `gpt2/model/`. To reuse an existing download, set `GPT2_MODEL_DIR` to that directory. No pretrained weights are committed. Loading the custom model happens when `gpt2.py` is imported; the prompt comparison also loads a reference model and therefore needs memory for both.

This is a learning implementation with separate Python attention-head calls and growing tensor caches, not an optimized serving engine. **Speed and memory benchmarks are still pending.** GPT-2 is a base completion model, not an instruction-tuned assistant; its text can be repetitive, inaccurate, or offensive. Matching the reference verifies computation, not factual quality.

## The addition experiment

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
gpt2/
  gpt2.py               Custom GPT-2 forward pass, weight loading, KV caching
  prompt.py             Interactive comparison against Hugging Face GPT-2
  download_model.py     Download pinned pretrained weights and tokenizer
  requirements.txt      Inference dependencies
addition/
  main.py               Addition training and evaluation
  kaggle_run.py         Remote job submission and output download
  runs/                 Recorded logs and exact submitted source
```

## Credits and reading

The starting architecture and learning path follow [Andrej Karpathy's GPT lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY). The corpus is [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare).

GPT-2 weights and tokenizer come from [openai-community/gpt2](https://huggingface.co/openai-community/gpt2). The reference is [Hugging Face Transformers](https://github.com/huggingface/transformers); the original model is [OpenAI GPT-2](https://github.com/openai/gpt-2). Pretrained assets retain their upstream terms.

Relevant arithmetic reading: [Teaching Arithmetic to Small Transformers](https://arxiv.org/abs/2307.03381) and [Understanding Addition in Transformers](https://arxiv.org/abs/2310.13121). These informed the investigation; this repository is a learning project, not a reproduction of their full experimental setups.

## Next

Benchmark cached versus uncached decoding: time to first token, tokens per second, and memory usage. Then batch attention heads into larger tensor operations and measure the effect. Longer-term: LoRA/QLoRA and inference profiling. For addition, controlled answer-order experiments and the zero-answer termination failure remain open.
