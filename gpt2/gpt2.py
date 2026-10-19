# imports
import torch
from torch import nn
from torch.nn import functional as F
import json
import os
from pathlib import Path

MODEL_DIR = Path(os.environ.get("GPT2_MODEL_DIR", Path(__file__).resolve().parent / "model"))
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

# for cpu
# Constants
ratio = 0.9  # spliting of training and testing
block_size = 1024  # what is the maximum context length for predictions?
batch_size = 64  # how many independent sequences will we process in parallel?
learning_rate = 1e-3  # for optimizer (xnew = xold - lr*grad)
eval_intervals = 500  # how much interval to wait to calc loss
max_iters = 10000  # how many times to update weights
eval_iters = 200  # average 200 batches per split each evaluation
n_embd = 768  # when a token is converted to n_embd dimension vector(vec embedding)
n_head = 12  # number of attention heads running in parallel inside ONE block
n_layer = 12  # number of blocks
dropout = 0  # each node gets dropoout probablity of getting dropped
# ------

print("device:", device)
print("---------")
print()

# # h.<block_number>.<component>.<subcomponent>.<weight/bias>
# with open("model.safetensors", "rb") as f:
#     header_size = int.from_bytes(f.read(8), "little")
#     header = json.loads(f.read(header_size))
#     print(sorted(list(header.keys())))

tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
vocab_size = tokenizer.get_vocab_size()


def encode(text):
    return tokenizer.encode(text).ids


def decode(ids):
    return tokenizer.decode(ids)


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
        q, k, v = self.qkv(x).split(self.head_size, dim=-1)
        past_length = 0
        if use_cache:
            if self.k_cache is not None:
                past_length = self.k_cache.shape[1]
                k = torch.cat([self.k_cache, k], dim=1)
                v = torch.cat([self.v_cache, v], dim=1)
            self.k_cache, self.v_cache = k, v

        wei = (q @ k.transpose(-2, -1)) * self.head_size**-0.5
        # Offset rows by the cached length; include all old + new key columns.
        mask = self.tril[past_length:past_length + T, :past_length + T]
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
        # there is slight devation from paper where in attention is all you need , in that
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
        if use_cache and (self.training or torch.is_grad_enabled() or target is not None):
            raise ValueError("Caching requires eval(), no_grad(), and no targets.")
        B, T = input_idx.shape
        past_length = self.cache_length() if use_cache else 0
        if T == 0 or past_length + T > block_size:
            raise ValueError("Input plus cache must contain 1 to block_size tokens.")
        if use_cache and past_length:
            cache = self.blocks[0].sa.heads[0].k_cache
            if cache.shape[0] != B or cache.device != input_idx.device:
                raise ValueError("Reset the cache before changing batch or device.")
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
            if target is not None:
                raise ValueError("last_token_only is for inference without targets.")
            x = x[:, -1:, :]
        logits = self.lm_head(x)  # (B,T,vocal_size)
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
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError("Provide a nonempty (B, T) prompt.")
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
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    idx_next = logits.argmax(dim=-1, keepdim=True)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx
        finally:
            self.reset_cache()  # Also clear state after an interrupted generation.
            self.train(was_training)


class LayerNorm1d:
    """layer norm normalizes each token vector embedding from ~0 to ~1 so it does not explode
    but its too restrictive so you introduct gamma and beta parameres that can be learned"""

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = xhat * self.gamma + self.beta
        return self.out


m = GPT2(vocab_size)

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


# Copy checkpoint tensors into our differently named layers.
# Track destinations so no learned parameter is accidentally left random.
loaded_parameters = set()


def copy_parameter(destination, source):
    if destination.shape != source.shape:
        raise ValueError(
            f"Weight shape mismatch: {destination.shape} vs {source.shape}"
        )
    destination.copy_(source)
    loaded_parameters.add(id(destination))


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

missing = [
    name for name, param in m.named_parameters() if id(param) not in loaded_parameters
]
if missing:
    raise RuntimeError(f"Parameters not loaded: {missing}")
assert m.lm_head.weight is m.token_embedding_table.weight
print("GPT-2 weights loaded successfully (all learned parameters).")
m.to(device)


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
