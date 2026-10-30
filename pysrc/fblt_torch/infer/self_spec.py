# ruff: noqa: E741
import torch

from ..verify import segment_with_trailing_closed


def generate_greedy_selfspec(
    model, entropy_lm, patcher_config, prompt_bytes, max_new_bytes, config, stats=None
):
    out = list(prompt_bytes)
    target = len(prompt_bytes) + max_new_bytes
    l = len(prompt_bytes)
    window_k = int(config.get("window_k", config.get("k", 4)))
    d0_mode = config.get("d0_mode", "zeros")
    while l < target:
        r_eff = min(window_k, target - l - 1)
        patches, trailing = segment_with_trailing_closed(entropy_lm, patcher_config, out, l)
        ids = torch.tensor(out[:l], dtype=torch.uint8)
        P, h = model.encode(ids, patches)
        g = model.global_model(P)
        if r_eff <= 0:
            logits = model.decoder(h, g, patches, trailing_closed=trailing)
            nxt = int(logits[l - 1].argmax().item()) if l > 0 else 0
            out.append(nxt)
            l += 1
            continue
        logits = model.decoder(h, g, patches, trailing_closed=trailing)
        draft = [int(logits[l - 1].argmax().item())]
        for m in range(1, r_eff):
            pos = l + m - 1
            d0_rows = torch.zeros(1, model.embed_dim)
            if d0_mode == "learned":
                table = getattr(model.diffusion_decoder, "d0_embed", None)
                if table is not None:
                    tok = draft[-1]
                    d0_rows = table.weight[tok].unsqueeze(0).detach().clone()
            bh = (
                torch.cat([h[:l], d0_rows.repeat(1, 1)], dim=0)
                if False
                else torch.cat([h[:l], d0_rows], dim=0)
            )
            # pad to pos+1 rows if needed
            if bh.shape[0] < pos + 1:
                bh = torch.cat([bh, torch.zeros(pos + 1 - bh.shape[0], model.embed_dim)], dim=0)
            logits2 = model.decoder(bh[: pos + 1], g, patches, trailing_closed=False)
            draft.append(int(logits2[pos].argmax().item()))
        out[l : l + r_eff] = draft[:r_eff]
        from .verify import verify_candidate

        verifier = verify_candidate
        l = verifier(model, entropy_lm, patcher_config, out, l, r_eff, target)
    return out[len(prompt_bytes) :]
