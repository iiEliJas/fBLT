import torch

from ..unmask import confidence_unmask


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
    still_masked = torch.ones(block_size, dtype=torch.bool)
    out_tokens = torch.full((block_size,), 256, dtype=torch.long)
    for _ in range(block_size):
        N = byte_hidden.shape[0]
        block_hidden = torch.zeros(block_size, byte_hidden.shape[1])
        full_hidden = torch.cat([byte_hidden, block_hidden], dim=0)
        pos = torch.arange(N, N + block_size)
        logits = decoder(full_hidden, patch_in, out_tokens, pos, patches)
        block_logits = logits[N:]
        probs = torch.softmax(block_logits, dim=-1)
        p_max, preds = probs.max(dim=-1)
        unmask = confidence_unmask(p_max, threshold, still_masked)
        for i in range(block_size):
            if unmask[i] and still_masked[i]:
                out_tokens[i] = preds[i].item()
                still_masked[i] = False
        if not still_masked.any():
            break
    return out_tokens
