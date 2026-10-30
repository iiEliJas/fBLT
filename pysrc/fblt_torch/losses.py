import torch
import torch.nn.functional as F


def l_clean(logits, targets, clean_sum=False, last_row_scale=1.0, has_window_plus_one=False):
    # logits: (N or N?, V); targets: (N+1,) bytes
    if has_window_plus_one:
        clean_rows = logits.shape[0]
        tgt = targets[1 : 1 + clean_rows]
        logits_used = logits
    else:
        clean_rows = logits.shape[0] - 1
        tgt = targets[1 : 1 + clean_rows]
        logits_used = logits[:clean_rows]
    ce = F.cross_entropy(logits_used, tgt.long(), reduction="mean")
    loss = ce
    if clean_sum:
        loss = loss * clean_rows
    if has_window_plus_one and last_row_scale != 1.0:
        sl = logits.shape[0]
        last_tgt = targets[sl : sl + 1].long()
        last_ce = F.cross_entropy(logits[-1:].contiguous(), last_tgt, reduction="mean")
        loss = loss + (last_row_scale - 1.0) * last_ce / logits.shape[0]
    return loss


def l_mask(logits_block_rows, targets_masked, cell_masked, t, loss_scale=1.0, mask_norm=False):
    # logits_block_rows: (R, V); targets_masked: (R,); cell_masked: (R,)
    masked_indices = [i for i in range(len(cell_masked)) if cell_masked[i]]
    if not masked_indices:
        return torch.tensor(0.0, device=logits_block_rows.device)
    total = 0.0
    for i in masked_indices:
        row = logits_block_rows[i]
        tgt = targets_masked[i]
        tgt_t = torch.tensor([tgt], device=row.device).long()
        total += F.cross_entropy(row.unsqueeze(0), tgt_t, reduction="sum").item()
    val = torch.tensor(total, device=logits_block_rows.device)
    if mask_norm:
        val = val / len(masked_indices)
    return loss_scale * val / t
