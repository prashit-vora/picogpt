# picogpt

A small character-level Transformer trained to generate Shakespeare-like dialogue. Built in PyTorch while following [Andrej Karpathy’s “Let’s build GPT: from scratch, in code, spelled out”](https://www.youtube.com/watch?v=kCc8FmEb1nY).

This is a learning project: the code includes my notes as I worked through embeddings, attention, residual connections, layer normalization, and dropout.

## Files

- `main.py` — model, training loop, evaluation, and text generation.
- `input.txt` — Shakespeare training text.
- `more.txt` — a sample of the generated output.

## Model

A decoder-only Transformer that predicts the next character:

```text
Character IDs → token + position embeddings
              → 6 Transformer blocks
              → final LayerNorm → vocabulary logits
```

Each block uses causal multi-head self-attention and a feedforward network, with pre-layer normalization and residual connections. Generation samples the next character from the predicted probabilities.

| Setting | Value |
| --- | --- |
| Context length | 256 characters |
| Embedding width | 384 |
| Attention heads per block | 6 |
| Transformer blocks | 6 |
| Feedforward hidden width | 1,536 |
| Dropout | 0.2 |
| Batch size | 64 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Training steps | 5,000 |
| Train / validation split | 90% / 10% |

Evaluation runs every 500 steps and averages 200 batches per split. The class is still named `BigramLangauageModel` from the early tutorial stage, but the current model uses attention over previous characters, not just a bigram lookup.

## Run

Run these commands from the repository directory:

```bash
git clone https://github.com/prashit-vora/picogpt.git
cd picogpt
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install torch numpy
python3 main.py
```

The script selects CUDA when available and otherwise uses CPU. A compatible CUDA-enabled PyTorch installation is needed for GPU training; the full configuration is slow on CPU.

After training, it generates 5,000 new characters, prints the text, and writes it to `more.txt` in the current directory. An existing `more.txt` is overwritten. Model weights are not currently saved, so each run starts training from scratch.

## Results

A completed Kaggle GPU run reported final training loss **1.0491** and validation loss **1.4912** (character-level cross-entropy). The final print statement currently labels this as `step 10000`, but the training loop uses `max_iters`, which is set to **5,000**.

An excerpt from `more.txt`:

```text
GLOUCESTER:
My gracious lord, my grave, and my Lord of Surrey,
Whom I was colour made in Edward's speech'd followers sweet,
And with me keep of flesh war, therefore I had princely ye,
```

The model picks up dialogue formatting, character names, and Shakespeare-like phrasing. Grammar and meaning are inconsistent, and it does not maintain a coherent story. These results are from one run, not a reproducible benchmark.

## Acknowledgments

Architecture and learning path follow [Andrej Karpathy’s nn-zero-to-hero lectures](https://github.com/karpathy/nn-zero-to-hero). The training corpus is Tiny Shakespeare, also available in [Karpathy’s character-RNN dataset collection](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare).

## Next

I will work on teaching the neural network addition.
