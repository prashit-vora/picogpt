import torch
from torch import nn
from torch.nn import functional as F
import json
from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent
from safetensors.torch import load_file
from tokenizers import Tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ratio = 0.9
block_size = 1024
batch_size = 4
learning_rate = 0.0001
eval_intervals = 500
max_iters = 10000
eval_iters = 200
n_embd = 768
n_head = 12
n_layer = 12
dropout = 0
print('device:', device)
print('---------')
print()
if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset('roneneldan/TinyStories', streaming=True)
    train_texts = [row['text'] for row in dataset['train'].shuffle(seed=42, buffer_size=10000).take(5000)]
    val_texts = [row['text'] for row in dataset['validation'].take(500)]
tokenizer = Tokenizer.from_file(str('/kaggle/temp/gpt2/tokenizer.json'))
vocab_size = tokenizer.get_vocab_size()

def encode(text):
    return tokenizer.encode(text).ids

def decode(ids):
    return tokenizer.decode(ids)
eos_id = tokenizer.token_to_id('<|endoftext|>')

def tokenize_stories(stories):
    ids = []
    for story in stories:
        ids.extend(encode(story))
        ids.append(eos_id)
    return torch.tensor(ids, dtype=torch.long)
if __name__ == '__main__':
    train_data = tokenize_stories(train_texts)
    val_data = tokenize_stories(val_texts)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return (x.to(device), y.to(device))
"\n\narchitecture:\n\nB: tot batch, T: context or block size, C: dimension of a vector (token-> vector); so you reprsent a token in a vector form (vec embedding)\ninput(B,T,C) -> vec embedding + pos embedding -> self-atttention(n times) -> output <- input(B,T,C) (this is called residual network)\n\nIn self-atttention:\n...-> (key,query,value) -> simple mlp -> (key,query,value) -> simple mlp -> .. ntimes ..\n\nIn residual network(ResNet):\nyou have this structure:\n        self-atttention \n        /                 input   ----------->  output            \n        (this is residual)\nwhy do we also connect input to output is because since this is like x = x + f(x)\nso we already have a big nn , will have a vanishing gradient problem it becomes extremly small large during backprop\nand big nn doesnot always imporve perfomannce it might degrade it sometime; \nso to solve this problem by haveing this x = x + f(x) so even if f'(x) -> 0 , you still have x' = 1 so you will have some gradient..\nand its also uselful for starting stages because in relu in starting stages (since weight are initlizaed randomly ) some will have 0\nand they might not be useful so its uselful to have imput to oputp and as the trainin inc the f(x) becomes more important...and it will be active.\n"

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.qkv = nn.Linear(n_embd, 3 * head_size)
        self.k_cache = None
        self.v_cache = None
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, use_cache=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.head_size, dim=-1)
        past_length = 0
        if use_cache:
            if self.k_cache is not None:
                past_length = self.k_cache.shape[1]
                k = torch.cat([self.k_cache, k], dim=1)
                v = torch.cat([self.v_cache, v], dim=1)
            self.k_cache, self.v_cache = (k, v)
        wei = q @ k.transpose(-2, -1) * self.head_size ** (-0.5)
        mask = self.tril[past_length:past_length + T, :past_length + T]
        wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ v

class MultiHeadAttention(nn.Module):
    """multiple heads in self-atttention in paraller but instead of doing one attention you group it and then add it"""

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
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU('tanh'), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """you will repeat the self-atttention and the mlp multiople times"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, use_cache=False):
        x = x + self.sa(self.ln1(x), use_cache=use_cache)
        x = x + self.ffwd(self.ln2(x))
        return x
'\nimplement the class \nforward(): return logits and loss based on idx and target\ngenrate(): based on the softmax genrate new max_new_tokens\n'

class GPT2(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight

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
        pos_emb = self.position_embedding_table(torch.arange(past_length, past_length + T, device=input_idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.ln_f(x)
        if last_token_only:
            x = x[:, -1:, :]
        logits = self.lm_head(x)
        B, T, C = logits.shape
        if target is None:
            loss = None
        else:
            logits = logits.reshape(B * T, C)
            target = target.reshape(B * T)
            loss = F.cross_entropy(logits, target, ignore_index=-100)
        return (logits, loss)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, use_cache=True, do_sample=True):
        """Prefill the prompt once, then process only each newly generated token."""
        was_training = self.training
        self.eval()
        self.reset_cache()
        try:
            for _ in range(max_new_tokens):
                if use_cache and 0 < self.cache_length() < block_size:
                    idx_cond = idx[:, -1:]
                else:
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
            self.reset_cache()
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
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.orignal(x) + self.scale * self.B(self.A(x))
m = GPT2(vocab_size)
weights = load_file(str('/kaggle/temp/gpt2/model.safetensors'))

def copy_parameter(destination, source):
    destination.copy_(source)
with torch.no_grad():
    for name, tensor in weights.items():
        parts = name.split('.')
        if parts[0] == 'wte':
            copy_parameter(m.token_embedding_table.weight, tensor)
        elif parts[0] == 'wpe':
            copy_parameter(m.position_embedding_table.weight, tensor)
        elif parts[0] == 'ln_f':
            copy_parameter(getattr(m.ln_f, parts[1]), tensor)
        elif parts[0] == 'h':
            layer = int(parts[1])
            node = parts[2]
            block = m.blocks[layer]
            if node == 'attn':
                if parts[3] == 'bias':
                    continue
                which, para = (parts[3], parts[4])
                if which == 'c_attn':
                    q, k, v = tensor.chunk(3, dim=-1)
                    for j, head in enumerate(block.sa.heads):
                        start = j * head.head_size
                        end = start + head.head_size
                        if para == 'weight':
                            source = torch.cat([q[:, start:end], k[:, start:end], v[:, start:end]], dim=1).T
                        elif para == 'bias':
                            source = torch.cat([q[start:end], k[start:end], v[start:end]])
                        copy_parameter(getattr(head.qkv, para), source)
                elif which == 'c_proj':
                    source = tensor.T if para == 'weight' else tensor
                    copy_parameter(getattr(block.sa.proj, para), source)
            elif node == 'mlp':
                which, para = (parts[3], parts[4])
                if which == 'c_fc':
                    linear = block.ffwd.net[0]
                elif which == 'c_proj':
                    linear = block.ffwd.net[2]
                source = tensor.T if para == 'weight' else tensor
                copy_parameter(getattr(linear, para), source)
            elif node in ('ln_1', 'ln_2'):
                norm = block.ln1 if node == 'ln_1' else block.ln2
                copy_parameter(getattr(norm, parts[3]), tensor)
print('GPT-2 weights loaded successfully (all learned parameters).')
print('----')
print()
m.requires_grad_(False)
for blocks in m.blocks:
    for head in blocks.sa.heads:
        head.qkv = LoRALinear(head.qkv, rank=8, alpha=8)
total = sum((p.numel() for p in m.parameters()))
trainable = sum((p.numel() for p in m.parameters() if p.requires_grad))
print(f'Total parameters: {total:,}')
print(f'Trainable parameters: {trainable:,}')
print('---')
print()
m.to(device)

@torch.no_grad()
def estimate_loss():
    losses = {}
    m.eval()
    for split in ('train', 'val'):
        batch_losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = m(xb, yb)
            batch_losses[i] = loss.item()
        losses[split] = batch_losses.mean().item()
    m.train()
    return losses
if __name__ == '__main__':
    optimizer = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], learning_rate)
    for epoch in range(max_iters):
        if epoch % eval_intervals == 0:
            losses = estimate_loss()
            print(f'step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')
        xb, yb = get_batch('train')
        logits, loss = m(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses = estimate_loss()
    print(f'step {max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')
    torch.save({'adapters': {name: param.detach().cpu() for name, param in m.named_parameters() if name.endswith(('.A.weight', '.B.weight'))}, 'config': {'rank': 8, 'alpha': 8}, 'training_completed': True}, MODEL_DIR / 'lora_adapters.pt')
    print('Saved', MODEL_DIR / 'lora_adapters.pt')
"\nso for attention is for x vector in bth batch i want it to depend from 1 to x-1 words for ex it will depend 10% from word1,1% w2...etc\nso you can like first have a weight matrix W(initializaed at random) and its dim will be .. Col will be C and row will be also C (since there are C tokens to work\n\nW = (T,T) = (8,8) = torch.tril(lower triangular)\n            a1 a2  a3  a4  a5  a6  a7  a8\n      a1  [[1., 0., 0., 0., 0., 0., 0., 0.],\n      a2   [1., 1., 0., 0., 0., 0., 0., 0.],\n      a3   [1., 1., 1., 0., 0., 0., 0., 0.],\n      a4   [1., 1., 1., 1., 0., 0., 0., 0.],\n      a5   [1., 1., 1., 1., 1., 0., 0., 0.],\n      a6   [1., 1., 1., 1., 1., 1., 0., 0.],\n      a7   [1., 1., 1., 1., 1., 1., 1., 0.],\n      a8   [1., 1., 1., 1., 1., 1., 1., 1.]])\nW @ X -> (8,8) @ (2,8,32) -> tensor will multiply with each batch -> (8,8) * (8,32)\nso for each token ai it will be avg of a1 to ai by multiplying W/w.sum(col) @ X -> so you are avging it and one way to modify the W-\nin W you can unmask - to float('-inf') first reason is i dont want x token to have influence form future token vectors and\nssec is inseatead of dividing by sum you can apply softmax to W matrix\ntensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],\n        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],\n        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],\n        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],\n        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],\n        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],\n        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],\n        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])\n\nnow the thing is each vector/token may not be depenednt upon all the past tokens\nso maybe avging over is oversimplfying is and sec reason is suppose a tokens wants \nto ask past tokens that i only want consonents or smthg so its a query...\n        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],\n        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],\n        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])\n\nnow the thing is each vector/token may not be depenednt upon all the past tokens\nso maybe avging over is oversimplfying is and sec reason is suppose a tokens wants \nto ask past tokens that i only want consonents or smthg so its a query...\nso each vector(token) will have a query to ask and a key(information to give)\nq = x @ wq ; wq -> (C/n_emb _>vector embeddinmg, head_size -> hyperparametere)\nk = x @ wk; same\nnow to check the affinity of which query is the key assoicated with\nso you will have a dot product k.q and see which query has high score\nand you take that vector embedding (x) or if insteda of x you will define a new\nvalue v = x @ wv where you will take value v (if its affinity/dot product ishigh)\n\n\nnn.Dropout() is to prevent overfititing when you scale the model, so what it does it that every forward and backward pass\nit shuts or zeros down some subset of nodes during training temporary\n"