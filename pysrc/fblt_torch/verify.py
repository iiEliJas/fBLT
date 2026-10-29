# ruff: noqa: E741
import torch

from .patching import next_starts_patch, segment_patches


def compute_entropy_vals(entropy_lm, byte_ids):
    with torch.no_grad():
        logits = entropy_lm(torch.tensor(byte_ids, dtype=torch.uint8))
        probs = torch.softmax(logits, dim=-1)
        ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    return ent.tolist()


def segment_with_trailing_closed(entropy_lm, patcher_config, bytes_seq, length):
    ent = compute_entropy_vals(entropy_lm, list(bytes_seq[:length]))
    patches = segment_patches(ent, list(bytes_seq[:length]), 1024, patcher_config)
    trailing = 0
    if patches:
        ls, ll = patches[-1]
        trailing = next_starts_patch(list(bytes_seq[:length]), length, ent, ls, ll, patcher_config)
    return patches, bool(trailing)


def verify_candidate(model, entropy_lm, patcher_config, x, l, r, target_len, stats=None):
    cand_len = l + r
    if cand_len > getattr(model, "max_seq_len", 10**9):
        raise ValueError("candidate too long")
    patches, trailing_closed = segment_with_trailing_closed(entropy_lm, patcher_config, x, cand_len)
    byte_ids = torch.tensor(x[:cand_len], dtype=torch.uint8)
    P, h = model.encode(byte_ids, patches)
    g = model.global_model(P)
    logits = model.decoder(h, g, patches, trailing_closed=trailing_closed)
    rows = logits.detach().cpu()
    for p in range(l, cand_len):
        pred = int(rows[p - 1].argmax().item())
        if x[p] != pred:
            x[p] = pred
            if stats is not None:
                stats["bytes_accepted"] = stats.get("bytes_accepted", 0) + (p - l)
            return p + 1
    if stats is not None:
        stats["bytes_accepted"] = stats.get("bytes_accepted", 0) + r
    if cand_len < target_len:
        x[cand_len] = int(rows[cand_len - 1].argmax().item())
        return cand_len + 1
    return cand_len
