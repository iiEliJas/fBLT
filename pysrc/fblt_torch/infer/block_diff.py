import torch

from ..unmask import confidence_unmask, entropy_unmask


def draft_block(
    decoder,
    byte_hidden,
    patch_in,
    patches,
    block_size,
    strategy="confidence",
    threshold=0.7,
    generator=None,
):
    N = byte_hidden.shape[0]
    # single block: clean rows N, block rows block_size
    still_masked = torch.ones(block_size, dtype=torch.bool)
    tokens = torch.full((block_size,), 256, dtype=torch.long)  # MASK
    # build a minimal batch-like layout on the fly
    out_tokens = tokens.clone()
    for _ in range(block_size):
        # construct byte_hidden_in for decoder: N clean + block_size rows (use zeros for block d0)
        block_hidden = torch.zeros(block_size, byte_hidden.shape[1])
        full_hidden = torch.cat([byte_hidden, block_hidden], dim=0)
        # patch_in expanded not handled; assume k=1
        logits = decoder(full_hidden, patch_in, patches)  # (N+block_size, V)
        block_logits = logits[N:]  # (block_size, V)
        probs = torch.softmax(block_logits, dim=-1)
        p_max, preds = probs.max(dim=-1)
        # entropy
        ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
        if strategy == "confidence":
            unmask = confidence_unmask(p_max, threshold, still_masked)
        else:
            unmask = entropy_unmask(ent, threshold, still_masked)
        # apply unmask
        for i in range(block_size):
            if unmask[i] and still_masked[i]:
                out_tokens[i] = preds[i].item()
                still_masked[i] = False
        if not still_masked.any():
            break
    return out_tokens
