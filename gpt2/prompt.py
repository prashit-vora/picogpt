import gpt2 as main
import torch
import time
from wonderwords import RandomWord

max_new_tokens = 100
tot_trial = 5
tot_time_use_cache_true = 0
tot_time_use_cache_false = 0

main.m.eval()

rw = RandomWord()
two_word = [f"{rw.word()} {rw.word()}" for _ in range(tot_trial)]

for use_cache in (False, True):
    prompt = two_word[0]
    idx = torch.tensor(
        [main.encode(prompt)],
        dtype=torch.long,
        device=main.device,
    )
    your_output = main.m.generate(
        idx, max_new_tokens=5, use_cache=use_cache, do_sample=False
    )

i = 0
while i < tot_trial:
    for use_cache in (False, True):
        prompt = two_word[i]
        if not use_cache:
            print(prompt)
        idx = torch.tensor(
            [main.encode(prompt)],
            dtype=torch.long,
            device=main.device,
        )

        if main.device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        your_output = main.m.generate(
            idx, max_new_tokens=max_new_tokens, use_cache=use_cache, do_sample=False
        )
        if main.device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        print(f"\nYOUR MODEL (use_cache={use_cache}):")
        print("---")
        print(main.decode(your_output[0].tolist()))

        tot_time = end_time - start_time
        if use_cache:
            tot_time_use_cache_true += tot_time
        else:
            tot_time_use_cache_false += tot_time

        print("---")
        print(f"Total time taken with use_cache = {use_cache} is {tot_time}")
        print("---")
        print()
    i += 1

mean_use_cache_true = tot_time_use_cache_true / tot_trial
mean_use_cache_false = tot_time_use_cache_false / tot_trial
print(f"Total time taken with use_cache = false is {tot_time_use_cache_false}")
print(f"Total time taken with use_cache = true is {tot_time_use_cache_true}")
print("---")
print(f"avg time taken per trial with use_cache = false is {mean_use_cache_false:.6f}")
print(f"avg time taken per trial with use_cache = true is {mean_use_cache_true:.6f}")

print(f"Speedup: {tot_time_use_cache_false / tot_time_use_cache_true:.2f}x")
