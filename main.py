import torch
from torch import nn
from torch.nn import functional as F

"""
architecture:
data -> decide batch_size for paraller processing -> split the data(80,10)
-> convert into tokens -> goes into the model for some epochs -> attention
-> test....

"""
# Constants
ratio = 0.9  # spliting of training and testing
block_size = 256  # what is the maximum context length for predictions?
batch_size = 64  # how many independent sequences will we process in parallel?
learning_rate = 3e-4  # for optimizer (xnew = xold - lr*grad)
eval_intervals = 500  # how much interval to wait to calc loss
max_iters = 5000  # how many times to update weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200  # average 200 batches per split each evaluation
n_embd = 384  # when a token is converted to n_embd dimension vector(vec embedding)
n_head = 6  # number of attention heads running in parallel inside ONE block
n_layer = 6  # number of blocks
dropout = 0.2  # each node gets dropoout probablity of getting dropped
# ------


print("device:", device)
print("---------")
print()

torch.manual_seed(1337)
# read the file
with open("input.txt", "r") as f:
    text = f.read()


# encoder and decoder
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda ids: "".join(itos[i] for i in ids)

# convert encoded data into tensor
data = torch.tensor(encode(text), dtype=torch.long, device=device)


# split it into training and testing (90: 10)
n = int(ratio * len(data))
train_data = data[:n]
val_data = data[n:]

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


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # (B,T,C)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads in self-atttention in paraller but instead of doing one attention you group it and then add it"""

    # TODO:: idk why is there a projection matrix ???
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
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
            nn.ReLU(),
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

    def forward(self, x):
        # there is slight devation from paper where in attention is all you need , in that
        # you will add layernorm after the sa and forward but there is slight devation since
        # you apply before both.
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# bigram model
"""
implement the class 
forward(): return logits and loss based on idx and target
genrate(): based on the softmax genrate new max_new_tokens
"""


class BigramLangauageModel(nn.Module):
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
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, input_idx, target=None):
        """lets say you have batchsize=2, context_window=3, chars = [[a,b,c], [a.c.e]] vocab = ['a,'b,'c','e'] NOTE: i know its a bigram mdoel so context_window > 2 doesnt matter but for future changes its coherent you convert chars to embedding (probs of what char will come next given a char) so you convert the chars to embedding to become idx
        idx -> (B,T,C) (in the ex : 2 batches , 3 headsize, 4 n_embed
        idx = [
               [[0,0,1,0],
               [1,0,0,0],
               [0,0,0,1]]
               ,
               [[0,0,1,0],
               [0,0,0,1]
               ,[1,0,0,0]]
             ]
        """
        B, T = input_idx.shape
        tok_emb = self.token_embedding_table(input_idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb
        # applying self attention
        x = self.blocks(x)
        # applying final layernorm
        x = self.ln_f(x)
        # prodcues logit
        logits = self.lm_head(x)  # (B,T,vocal_size)
        B, T, C = logits.shape
        if target is None:
            loss = None
        else:
            logits = logits.reshape(B * T, C)
            target = target.reshape(B * T)
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # you will append a token by predicting it in forward so you only take the last block size
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


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


# print(
#     decode(
#         m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[
#             0
#         ].tolist()
#     )
# )

m = BigramLangauageModel(vocab_size)
m.to(device)
optimizer = torch.optim.AdamW(m.parameters(), learning_rate)


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
print(f"step 10000: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# generate from the model
m.eval()
with torch.no_grad():
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(m.generate(context, max_new_tokens=5000)[0].tolist())
    print(generated_text)
    with open("more.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)

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
