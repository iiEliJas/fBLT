import torch

from ..patching import compute_entropy_vals, segment_patches


def generate_greedy(model, entropy_lm, patcher_config, prompt_bytes, new_bytes):
    out = list(prompt_bytes)
    target = len(prompt_bytes) + new_bytes
    while len(out) < target:
        patches = segment_patches(compute_entropy_vals(entropy_lm, out), out, 1024, patcher_config)
        ids = torch.tensor(out, dtype=torch.uint8)
        P, h = model.encode(ids, patches)
        g = model.global_model(P)
        logits = model.decoder(h, g, patches)
        nxt = int(logits[-1].argmax().item())
        out.append(nxt)
    return out[len(prompt_bytes) :]


def generate_blockdiff(
    model, entropy_lm, patcher_config, prompt_bytes, new_bytes, block_size=4, config=None
):
    from ..blocks import build_block_batch

    if config is None:
        config = {}
    out = list(prompt_bytes)
    target = len(prompt_bytes) + new_bytes
    while len(out) < target:
        patches = segment_patches(compute_entropy_vals(entropy_lm, out), out, 1024, patcher_config)
        B = min(block_size, target - len(out))
        ids = torch.tensor(out, dtype=torch.uint8)
        P, h = model.encode(ids, patches)
        g = model.global_model(P)
        batch = build_block_batch(out, patches, B, 1, t_in=0.5)
        batch["get"] = lambda k, d=None: batch.get(k, d)
        logits, _ = model.forward_diffusion(h, g, patches, batch, out, None, trailing_closed=True)
        block_logits = logits[len(out) :]
        block_tokens = block_logits.argmax(dim=-1).tolist()
        for tok in block_tokens:
            if len(out) >= target:
                break
            out.append(int(tok))
    return out[len(prompt_bytes) :]
