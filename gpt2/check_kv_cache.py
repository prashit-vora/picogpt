"""Run with python3 check_kv_cache.py; compares cached and full attention."""
import torch

torch.set_num_threads(4)
import gpt2 as main

m = main.m.eval()

@torch.no_grad()
def check_chunks(model, ids, chunks):
    model.reset_cache()
    expected = model(ids)[0]
    results = []
    offset = 0
    for length in chunks:
        results.append(model(ids[:, offset:offset + length], use_cache=True)[0])
        offset += length
    assert offset == ids.shape[1]
    actual = torch.cat(results, dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-4)
    # An ordinary forward must ignore and preserve existing cache.
    torch.testing.assert_close(model(ids)[0], expected, atol=1e-3, rtol=1e-4)
    assert model.cache_length() == ids.shape[1]
    model.reset_cache()
    print('PASS chunks', chunks, 'max difference', (actual-expected).abs().max().item())

ids = torch.tensor([main.encode('The capital of France is'),
                    main.encode('The capital of Japan is')], device=main.device)
check_chunks(m, ids, [ids.shape[1]-1, 1])
check_chunks(m, ids, [1]*ids.shape[1])
check_chunks(m, ids, [2, ids.shape[1]-2])

cached = m.generate(ids, 8, use_cache=True, do_sample=False)
uncached = m.generate(ids, 8, use_cache=False, do_sample=False)
assert torch.equal(cached, uncached)
assert m.cache_length() == 0
assert torch.equal(m.generate(ids[:1], 8, do_sample=False), cached[:1])
print('PASS pretrained greedy output and fresh-prompt reset')

# Exercise window overflow cheaply with a small instance of the SAME classes.
main.block_size, main.n_embd, main.n_head, main.n_layer = 8, 24, 3, 2
small = main.GPT2(32).eval()
short = torch.tensor([[1,2,3,4,5,6,7], [7,6,5,4,3,2,1]])
check_chunks(small, short, [2,3,2])
assert torch.equal(small.generate(short,5,do_sample=False),
                   small.generate(short,5,use_cache=False,do_sample=False))
long = torch.tensor([[1,2,3,4,5,6,7,8,9,10]])
assert torch.equal(small.generate(long,3,do_sample=False),
                   small.generate(long,3,use_cache=False,do_sample=False))
try:
    with torch.no_grad():
        small(short, use_cache=True)
        small(short, use_cache=True)
except ValueError:
    pass
else:
    raise AssertionError('Cache overflow must raise')
finally:
    small.reset_cache()
print('PASS sliding context, long prompt, and overflow guard')
print('All KV cache checks passed.')
