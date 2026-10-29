# ruff: noqa: E741
import torch

from ..unmask import confidence_unmask, entropy_unmask


def draft_block(
    decoder, byte_hidden, patch_in, patches, prefix_len, block_size, config, rng_state=None
):
    if rng_state is None:
        rng_state = [config.get("seed", 1) or 1]
    B = block_size
    tokens = [256] * B
    masked = [True] * B
    passes = 0
    threshold = config.get("threshold", 0.7)
    strategy = config.get("strategy", "confidence")
    for _ in range(B):
        if not any(masked):
            break
        batch = {
            "num_clean": prefix_len,
            "block_size": B,
            "num_blocks": 1,
            "n_block_rows": B,
            "t": 0.0,
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "positions": torch.tensor([prefix_len + b for b in range(B)], dtype=torch.long),
            "targets": torch.tensor([0] * B, dtype=torch.uint8),
            "cell_valid": torch.tensor([1] * B, dtype=torch.uint8),
            "cell_masked": torch.tensor([1 if m else 0 for m in masked], dtype=torch.uint8),
            "groups": torch.tensor([len(patches) - 1] * B, dtype=torch.long),
            "loss_scale": 0.0,
            "last_row_scale": 1.0,
            "clean_sum": False,
            "mask_norm": False,
        }
        # attach dict-like get
        batch["get"] = lambda k, d=None: batch.get(k, d)
        logits, _ = decoder(byte_hidden, patch_in, patches, batch, None, None, trailing_closed=True)
        block_logits = logits[prefix_len:]
        probs = torch.softmax(block_logits, dim=-1)
        p_max, preds = probs.max(dim=-1)
        if strategy == "confidence":
            conf = p_max.tolist()
            mmask = torch.tensor(masked, dtype=torch.bool)
            sel = confidence_unmask(torch.tensor(conf), threshold, mmask)
        else:
            ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
            mmask = torch.tensor(masked, dtype=torch.bool)
            sel = entropy_unmask(ent, threshold, mmask)
        for b in range(B):
            if sel[b] and masked[b]:
                tokens[b] = int(preds[b].item())
                masked[b] = False
        passes += 1
        if not any(masked):
            break
    out = [int(t) if t < 256 else 0 for t in tokens]
    return out, passes
