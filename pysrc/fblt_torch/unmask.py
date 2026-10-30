import torch


def confidence_unmask(p_max, threshold, still_masked):
    # p_max: (R,) ; still_masked: (R,) bool
    candidates = still_masked & (p_max >= threshold)
    if candidates.any():
        return candidates
    # unmask at least one: highest p_max among still masked
    idx = torch.where(still_masked)[0]
    if idx.numel() == 0:
        return still_masked
    best = idx[p_max[idx].argmax()]
    out = torch.zeros_like(still_masked)
    out[best] = True
    return out


def entropy_unmask(entropies, gamma, still_masked):
    idx = torch.where(still_masked)[0]
    if idx.numel() == 0:
        return still_masked
    order = idx[torch.argsort(entropies[idx])]
    cum = 0.0
    chosen = []
    for i in order.tolist():
        if cum + entropies[i].item() <= gamma:
            cum += entropies[i].item()
            chosen.append(i)
        else:
            break
    if not chosen:
        chosen = [order[0].item()]
    out = torch.zeros_like(still_masked)
    out[torch.tensor(chosen)] = True
    return out
