# imports
import torch
from torch import nn
from torch.nn import functional as F
import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
from safetensors.torch import load_file
from tokenizers import Tokenizer
# ----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for gpu
# Constants
# ratio = 0.9  # spliting of training and testing
# block_size = 256  # what is the maximum context length for predictions?
# batch_size = 64  # how many independent sequences will we process in parallel?
# learning_rate = 3e-4  # for optimizer (xnew = xold - lr*grad)
# eval_intervals = 500  # how much interval to wait to calc loss
# max_iters = 5000  # how many times to update weights
# eval_iters = 200  # average 200 batches per split each evaluation
# n_embd = 384  # when a token is converted to n_embd dimension vector(vec embedding)
# n_head = 6  # number of attention heads running in parallel inside ONE block
# n_layer = 6  # number of blocks
# dropout = 0.2  # each node gets dropoout probablity of getting dropped
# ------

# for gpt2
# Constants
ratio = 0.9  # spliting of training and testing
block_size = 1024  # what is the maximum context length for predictions?
batch_size = 4  # how many independent sequences will we process in parallel?
learning_rate = 1e-4  # for optimizer (xnew = xold - lr*grad)
eval_intervals = 500  # how much interval to wait to calc loss
max_iters = 10000  # how many times to update weights
eval_iters = 200  # average 200 batches per split each evaluation
n_embd = 768  # when a token is converted to n_embd dimension vector(vec embedding)
n_head = 12  # number of attention heads running in parallel inside ONE block
n_layer = 12  # number of blocks
dropout = 0  # each node gets dropoout probablity of getting dropped
temprature = 0.8
# ------

print("device:", device)
print("---------")
print()

# # h.<block_number>.<component>.<subcomponent>.<weight/bias>
# with open("model.safetensors", "rb") as f:
#     header_size = int.from_bytes(f.read(8), "little")
#     header = json.loads(f.read(header_size))
#     print(sorted(list(header.keys())))
#

if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("roneneldan/TinyStories", streaming=True)

    train_texts = [
        row["text"]
        for row in dataset["train"].shuffle(seed=42, buffer_size=10_000).take(5_000)
    ]  # 5000 stores to predict

    val_texts = [
        row["text"] for row in dataset["validation"].take(500)
    ]  # 500 stories for validation

tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
vocab_size = tokenizer.get_vocab_size()


def encode(text):
    return tokenizer.encode(text).ids


def decode(ids):
    return tokenizer.decode(ids)


eos_id = tokenizer.token_to_id("<|endoftext|>")


def tokenize_stories(stories):
    ids = []
    for story in stories:
        ids.extend(encode(story))
        ids.append(eos_id)  # marks the end of this story.
    return torch.tensor(ids, dtype=torch.long)


if __name__ == "__main__":
    train_data = tokenize_stories(train_texts)
    val_data = tokenize_stories(val_texts)


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


"""

architecture:

B: tot batch, T: context or block size, C: dimension of a vector (token-> vector); so you reprsent a token in a vector form (vec embedding)
input(B,T,C) -> vec embedding + pos embedding -> self-atttention(n times) -> output <- input(B,T,C) (this is called residual network)

In self-atttention:
...-> (key,query,value) -> simple mlp -> (key,query,value) -> simple mlp -> .. ntimes ..

In residual network(ResNet):
you have this structure:
        self-atttention
        /               \
  input   ----------->  output
        (this is residual)
why do we also connect input to output is because since this is like x = x + f(x)
so we already have a big nn , will have a vanishing gradient problem it becomes extremly small large during backprop
and big nn doesnot always imporve perfomannce it might degrade it sometime;
so to solve this problem by haveing this x = x + f(x) so even if f'(x) -> 0 , you still have x' = 1 so you will have some gradient..
and its also uselful for starting stages because in relu in starting stages (since weight are initlizaed randomly ) some will have 0
and they might not be useful so its uselful to have imput to oputp and as the trainin inc the f(x) becomes more important...and it will be active.
"""


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        # self.key = nn.Linear(n_embd, head_size)
        # self.query = nn.Linear(n_embd, head_size)
        # self.value = nn.Linear(n_embd, head_size)
        #
        # in gpt2 model they combnine all thre in one layer
        self.qkv = nn.Linear(n_embd, 3 * head_size)
        self.k_cache = None
        self.v_cache = None
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    # KV CACHE: each head stores past keys and values for one generation.
    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, use_cache=False):
        B, T, C = x.shape  # T is the prompt length, then usually 1.
        q, k, v = self.qkv(x).split(
            self.head_size, dim=-1
        )  # only the calc of new token not old
        past_length = 0
        if use_cache:
            if self.k_cache is not None:
                past_length = self.k_cache.shape[1]
                k = torch.cat([self.k_cache, k], dim=1)
                v = torch.cat([self.v_cache, v], dim=1)
            self.k_cache, self.v_cache = k, v

        wei = (q @ k.transpose(-2, -1)) * self.head_size**-0.5
        # offset rows by the cached length; include all old + new key columns.
        # so it will select the new token and all the past key
        mask = self.tril[past_length : past_length + T, : past_length + T]
        wei = wei.masked_fill(mask == 0, float("-inf"))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ v


class MultiHeadAttention(nn.Module):
    """multiple heads in self-atttention in paraller but instead of doing one attention you group it and then add it"""

    # TODO:: idk why is there a projection matrix ???
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_cache=False):
        out = torch.cat([h(x, use_cache=use_cache) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        # the 4 is a multiple factor define in the paper itself (attention is all you need)
        # to make more compute and make the network wider, the reason its 4 idk? its a hyperparametere
        # you select based on expirement
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            # TODO: gpt2 model uses gelu but idk why
            # its like x * phi(x); where phi(x) is a probablity being less than x
            nn.GELU("tanh"),
            nn.Linear(4 * n_embd, n_embd),  # proj matrix
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """you will repeat the self-atttention and the mlp multiople times"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        # n_head = number of attention heads running in parallel inside ONE block
        # head_size = number of features each head outputs
        # Example: n_embd=32, n_head=8 -> head_size=4
        # Concatenating 8 head outputs gives 8*4=32 features
        # Number of stacked blocks is chosen separately
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, use_cache=False):
        # you will add layernorm after the sa and forward but there is slight devation since
        # you apply before both.
        x = x + self.sa(self.ln1(x), use_cache=use_cache)
        x = x + self.ffwd(self.ln2(x))
        return x


# bigram model
"""
implement the class
forward(): return logits and loss based on idx and target
genrate(): based on the softmax genrate new max_new_tokens
"""


class GPT2(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # head_size = n_embd
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layernorm
        # after attention it will have language model producing logits
        # in gpt2 arch the weights of token_embedding_table and lm_head are same
        # it saves memory and computation as it works both as token embedding
        # and producing logits; and the token embedding works by W[token_id]
        # and mat mul woks in nn.Linear by x.W(T) so it will transpose the matrix
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight

    # KV CACHE: clear all layers before a new prompt.
    def reset_cache(self):
        for block in self.blocks:
            for head in block.sa.heads:
                head.reset_cache()

    def cache_length(self):
        cache = self.blocks[0].sa.heads[0].k_cache
        return 0 if cache is None else cache.shape[1]

    def forward(self, input_idx, target=None, use_cache=False, last_token_only=False):
        B, T = input_idx.shape
        past_length = self.cache_length() if use_cache else 0
        if use_cache and past_length:
            cache = self.blocks[0].sa.heads[0].k_cache
        tok_emb = self.token_embedding_table(input_idx)
        pos_emb = self.position_embedding_table(
            torch.arange(past_length, past_length + T, device=input_idx.device)
        )  # (T,C)
        x = tok_emb + pos_emb
        # applying self attention
        # Sequential cannot pass use_cache automatically: forward it to each block.
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        # applying final layernorm
        x = self.ln_f(x)
        # prodcues logit
        if last_token_only:
            x = x[:, -1:, :]
        logits = self.lm_head(x)  # (B,T,vocab_size)
        B, T, C = logits.shape
        if target is None:
            loss = None
        else:
            logits = logits.reshape(B * T, C)
            target = target.reshape(B * T)
            loss = F.cross_entropy(logits, target, ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, use_cache=True, do_sample=True):
        """Prefill the prompt once, then process only each newly generated token."""
        was_training = self.training
        self.eval()
        self.reset_cache()
        try:
            for _ in range(max_new_tokens):
                if use_cache and 0 < self.cache_length() < block_size:
                    idx_cond = idx[:, -1:]  # Decode: only the new token.
                else:
                    # Prefill, or rebuild after the context window slides.
                    # Learned position IDs restart for the cropped window, so
                    # simply dropping the oldest cached K/V would be incorrect.
                    self.reset_cache()
                    idx_cond = idx[:, -block_size:]
                logits, _ = self(idx_cond, use_cache=use_cache, last_token_only=True)
                logits = logits[:, -1, :]
                if do_sample:
                    top_logits, top_ids = torch.topk(logits, k=40, dim=-1)

                    probs = F.softmax(top_logits / temprature, dim=-1)
                    choice = torch.multinomial(probs, num_samples=1)
                    idx_next = top_ids.gather(dim=-1, index=choice)
                else:
                    idx_next = logits.argmax(dim=-1, keepdim=True)
                # stop genrating if you are at the end of story
                if idx_next.item() == eos_id:
                    break
                idx = torch.cat((idx, idx_next), dim=1)
            return idx
        finally:
            self.reset_cache()  # Also clear state after an interrupted generation.
            self.train(was_training)


class LoRALinear(nn.Module):
    """
    for now we will add LoRA to the qkv layer
    you can add it to any learnable weight
    but the most imp is qkv layer in this so
    for learning purpose ill add it to onlly qkv layer
    so ... input -> qkv layer + LoRA -> output ...
    qkv (n_embd, 3*head_size) so LoRA W = qkv also have the same dim
    and delW will split into matrix with dim (n_embd, r(rank)) and (r(rank), 3*head_size)
    a layer computes,  y = W0x ;(W0 -> pretrained qkv matrix)
    then you change W = W0 + delW
    and y  = W0x + delWx
    and that delW = BA
    y = W0x + BAx
    y = W0x + Bz (z = Ax)
    """

    def __init__(self, orignal, rank=8, alpha=8):
        super().__init__()
        self.orignal = orignal
        self.A = nn.Linear(orignal.in_features, rank, bias=False)
        self.B = nn.Linear(rank, orignal.out_features, bias=False)

        # initally B = 0, so its same as y = W0x
        nn.init.zeros_(self.B.weight)
        # TODO: this is a hyperparametre we add idk whats the purpose
        self.scale = alpha / rank

    def forward(self, x):
        return self.orignal(x) + self.scale * self.B(self.A(x))


m = GPT2(vocab_size)


# generate from the model
# m.eval()
# with torch.no_grad():
#     context = torch.zeros((1, 1), dtype=torch.long, device=device)
#     generated_text = decode(m.generate(context, max_new_tokens=5000)[0].tolist())
#     print(generated_text)
#     with open("more.txt", "w", encoding="utf-8") as f:
#         f.write(generated_text)
#

weights = load_file(str(MODEL_DIR / "model.safetensors"))

# h.0.attn.bias (1, 1, 1024, 1024) -> causal masking
# h.0.attn.c_attn.bias (2304,) -> qkv bias
# h.0.attn.c_attn.weight (768, 2304) -> qkv weight
# h.0.attn.c_proj.bias (768,) -> after qkv projection matrix bias
# h.0.attn.c_proj.weight (768, 768) -> after qkv projection matrix weight
# h.0.ln_1.bias (768,)  -> layernorm bias before qkv
# h.0.ln_1.weight (768,)-> layernorm weight before qkv
# h.0.ln_2.bias (768,)   -> layernorm bias before mlp
# h.0.ln_2.weight (768,) -> layernorm weight before mlp
# h.0.mlp.c_fc.bias (3072,) -> simple mlp bias after qkv
# h.0.mlp.c_fc.weight (768, 3072) -> simple mlp weight after qkv
# h.0.mlp.c_proj.bias (768,) -> after mlp projection matrix bias
# h.0.mlp.c_proj.weight (3072, 768)-> after mlp projection matrix weight


def copy_parameter(destination, source):
    destination.copy_(source)


with torch.no_grad():
    for name, tensor in weights.items():
        parts = name.split(".")

        if parts[0] == "wte":
            copy_parameter(m.token_embedding_table.weight, tensor)
        elif parts[0] == "wpe":
            copy_parameter(m.position_embedding_table.weight, tensor)
        elif parts[0] == "ln_f":
            copy_parameter(getattr(m.ln_f, parts[1]), tensor)
        elif parts[0] == "h":
            layer = int(parts[1])
            node = parts[2]
            block = m.blocks[layer]
            if node == "attn":
                if parts[3] == "bias":
                    continue  # each head already has its causal tril mask.

                which, para = parts[3], parts[4]
                if which == "c_attn":
                    # saved layout: [all Q | all K | all V].
                    q, k, v = tensor.chunk(3, dim=-1)
                    for j, head in enumerate(block.sa.heads):
                        start = j * head.head_size
                        end = start + head.head_size
                        if para == "weight":
                            source = torch.cat(
                                [
                                    q[:, start:end],
                                    k[:, start:end],
                                    v[:, start:end],
                                ],
                                dim=1,
                            ).T
                        elif para == "bias":
                            source = torch.cat(
                                [q[start:end], k[start:end], v[start:end]]
                            )
                        copy_parameter(getattr(head.qkv, para), source)
                elif which == "c_proj":
                    source = tensor.T if para == "weight" else tensor
                    copy_parameter(getattr(block.sa.proj, para), source)
            elif node == "mlp":
                which, para = parts[3], parts[4]
                if which == "c_fc":
                    linear = block.ffwd.net[0]
                elif which == "c_proj":
                    linear = block.ffwd.net[2]
                source = tensor.T if para == "weight" else tensor
                copy_parameter(getattr(linear, para), source)

            elif node in ("ln_1", "ln_2"):
                norm = block.ln1 if node == "ln_1" else block.ln2
                copy_parameter(getattr(norm, parts[3]), tensor)


print("GPT-2 weights loaded successfully (all learned parameters).")
print("----")
print()

m.requires_grad_(False)

# setting qkv = LoRALinear
for blocks in m.blocks:
    for head in blocks.sa.heads:
        head.qkv = LoRALinear(head.qkv, rank=8, alpha=8)

total = sum(p.numel() for p in m.parameters())
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"Total parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")
print("---")
print()

m.to(device)


@torch.no_grad()
def estimate_loss():
    losses = {}
    m.eval()
    for split in ("train", "val"):
        batch_losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = m(xb, yb)
            batch_losses[i] = loss.item()
        losses[split] = batch_losses.mean().item()
    m.train()
    return losses


if __name__ == "__main__":
    optimizer = torch.optim.AdamW(
        [p for p in m.parameters() if p.requires_grad], learning_rate
    )

    for epoch in range(max_iters):
        if epoch % eval_intervals == 0:
            losses = estimate_loss()
            print(
                f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch("train")
        logits, loss = m(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses = estimate_loss()
    print(
        f"step {max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
    )

    # Save only learned adapters; pretrained weights remain in model.safetensors.
    torch.save(
        {
            "adapters": {
                name: param.detach().cpu()
                for name, param in m.named_parameters()
                if name.endswith((".A.weight", ".B.weight"))
            },
            "config": {"rank": 8, "alpha": 8},
            "training_completed": True,
        },
        MODEL_DIR / "lora_adapters.pt",
    )
    print("Saved", MODEL_DIR / "lora_adapters.pt")

"""
so for attention is for x vector in bth batch i want it to depend from 1 to x-1 words for ex it will depend 10% from word1,1% w2...etc
so you can like first have a weight matrix W(initializaed at random) and its dim will be .. Col will be C and row will be also C (since there are C tokens to work

W = (T,T) = (8,8) = torch.tril(lower triangular)
            a1 a2  a3  a4  a5  a6  a7  a8
      a1  [[1., 0., 0., 0., 0., 0., 0., 0.],
      a2   [1., 1., 0., 0., 0., 0., 0., 0.],
      a3   [1., 1., 1., 0., 0., 0., 0., 0.],
      a4   [1., 1., 1., 1., 0., 0., 0., 0.],
      a5   [1., 1., 1., 1., 1., 0., 0., 0.],
      a6   [1., 1., 1., 1., 1., 1., 0., 0.],
      a7   [1., 1., 1., 1., 1., 1., 1., 0.],
      a8   [1., 1., 1., 1., 1., 1., 1., 1.]])
W @ X -> (8,8) @ (2,8,32) -> tensor will multiply with each batch -> (8,8) * (8,32)
so for each token ai it will be avg of a1 to ai by multiplying W/w.sum(col) @ X -> so you are avging it and one way to modify the W-
in W you can unmask - to float('-inf') first reason is i dont want x token to have influence form future token vectors and
ssec is inseatead of dividing by sum you can apply softmax to W matrix
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])

now the thing is each vector/token may not be depenednt upon all the past tokens
so maybe avging over is oversimplfying is and sec reason is suppose a tokens wants
to ask past tokens that i only want consonents or smthg so its a query...
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])

now the thing is each vector/token may not be depenednt upon all the past tokens
so maybe avging over is oversimplfying is and sec reason is suppose a tokens wants
to ask past tokens that i only want consonents or smthg so its a query...
so each vector(token) will have a query to ask and a key(information to give)
q = x @ wq ; wq -> (C/n_emb _>vector embeddinmg, head_size -> hyperparametere)
k = x @ wk; same
now to check the affinity of which query is the key assoicated with
so you will have a dot product k.q and see which query has high score
and you take that vector embedding (x) or if insteda of x you will define a new
value v = x @ wv where you will take value v (if its affinity/dot product ishigh)


nn.Dropout() is to prevent overfititing when you scale the model, so what it does it that every forward and backward pass
it shuts or zeros down some subset of nodes during training temporary
"""
