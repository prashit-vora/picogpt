import gpt2 as main
import torch
from transformers import GPT2LMHeadModel

# evaluate the model interactively
reference = GPT2LMHeadModel.from_pretrained(
    str(main.MODEL_DIR),
    local_files_only=True,
    attn_implementation="eager",
).to(main.device)

reference.eval()
main.m.eval()

while True:
    try:
        prompt = input("Prompt (blank to quit): ")
    except (EOFError, KeyboardInterrupt):
        break
    if not prompt.strip():
        break
    idx = torch.tensor(
        [main.encode(prompt)],
        dtype=torch.long,
        device=main.device,
    )

    with torch.no_grad():
        your_logits, _ = main.m(idx)
        reference_logits = reference(idx).logits

    print("Your shape:", your_logits.shape)
    print("Reference shape:", reference_logits.shape)

    difference = (your_logits - reference_logits).abs()
    print("Maximum difference:", difference.max().item())
    print("Mean difference:", difference.mean().item())

    torch.testing.assert_close(
        your_logits,
        reference_logits,
        atol=1e-3,
        rtol=1e-4,
    )
    print("PASS: logits match within tolerance")

    @torch.no_grad()
    def generate_greedy(model, idx, is_reference, max_new_tokens=30):
        idx = idx.clone()

        for _ in range(max_new_tokens):
            context = idx[:, -main.block_size :]

            if is_reference:
                logits = model(context).logits
            else:
                logits, _ = model(context)

            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, next_id), dim=1)

        return idx

    your_output = main.m.generate(
        idx, max_new_tokens=30, use_cache=True, do_sample=False
    )
    reference_output = generate_greedy(reference, idx, is_reference=True)
    print("\nYOUR MODEL (KV CACHE):")
    print(main.decode(your_output[0].tolist()))

    print("\nREFERENCE GPT-2:")
    print(main.decode(reference_output[0].tolist()))
    print("\nIdentical token IDs:", torch.equal(your_output, reference_output))
