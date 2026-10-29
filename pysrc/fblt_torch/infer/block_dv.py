# ruff: noqa: E741
import torch

from ..patching import compute_entropy_vals, next_starts_patch, segment_patches
from ..verify import verify_candidate
from .block_diff import draft_block


def _segment(entropy_lm, patcher_config, bytes_seq, length):
    ent = compute_entropy_vals(entropy_lm, bytes_seq[:length])
    patches = segment_patches(ent, bytes_seq[:length], 1024, patcher_config)
    return patches, ent


def _ar_next_byte(model, patches, bytes_seq, length):
    ids = torch.tensor(bytes_seq[:length], dtype=torch.uint8)
    P, h = model.encode(ids, patches)
    g = model.global_model(P)
    logits = model.decoder(h, g, patches, trailing_closed=False)
    return int(logits[length - 1].argmax().item())


def generate_common(
    model, entropy_lm, patcher_config, prompt_bytes, max_new_bytes, config, do_verify, stats=None
):
    out = list(prompt_bytes)
    target = len(prompt_bytes) + max_new_bytes
    l = len(prompt_bytes)
    adaptive = bool(do_verify and config.get("B_max", 0) and config.get("adapt_window", 0))
    cur_B = int(config.get("block_size", 4))
    acc_hist = []
    hist_len = 0
    adapt_window = int(config.get("adapt_window", 8))
    boundary_aligned = bool(config.get("boundary_aligned", False))
    while l < target:
        B = min(cur_B, target - l)
        config = dict(config)
        config["block_size"] = B
        patches, ent = _segment(entropy_lm, patcher_config, out, l)
        open_start = patches[-1][0]
        open_len = l - open_start
        starts_here = next_starts_patch(out, l, ent, open_start, open_len, patcher_config)
        if not starts_here:
            out.append(_ar_next_byte(model, patches, out, l))
            if stats is not None:
                stats["nfe_encoder_global"] = stats.get("nfe_encoder_global", 0) + 1
                stats["nfe_decoder"] = stats.get("nfe_decoder", 0) + 1
                stats["bytes_ar"] = stats.get("bytes_ar", 0) + 1
            l += 1
            continue
        ids = torch.tensor(out[:l], dtype=torch.uint8)
        P, h = model.encode(ids, patches)
        g = model.global_model(P)
        if stats is not None:
            stats["nfe_encoder_global"] = stats.get("nfe_encoder_global", 0) + 1
        draft, passes = draft_block(
            model.diffusion_decoder,
            h,
            g,
            patches,
            l,
            B,
            config,
        )
        if stats is not None:
            stats["nfe_decoder"] = stats.get("nfe_decoder", 0) + passes
            stats["bytes_drafted"] = stats.get("bytes_drafted", 0) + B
        out[l : l + B] = draft[:B]
        if not do_verify:
            if stats is not None:
                stats["bytes_accepted"] = stats.get("bytes_accepted", 0) + B
            l += B
        else:
            verifier = model
            if boundary_aligned:
                from ..verify import verify_candidate as _vc

                committed = _vc(verifier, entropy_lm, patcher_config, out, l, B, target)
            else:
                committed = verify_candidate(
                    verifier, entropy_lm, patcher_config, out, l, B, target
                )
            if stats is not None:
                pass
            if adaptive:
                acc = (committed - l) / B
                acc = min(1.0, max(0.0, acc))
                acc_hist.append(acc)
                if len(acc_hist) > adapt_window:
                    acc_hist.pop(0)
                hist_len = len(acc_hist)
                mean = sum(acc_hist) / hist_len
                # adaptive rule
                cur_B = int(cur_B)
                b_min = int(config.get("B_min", 1) or 1)
                b_max = int(config.get("B_max", 0) or 0)
                target_acc = float(config.get("accept_target", 0.5))
                if b_max and b_max >= b_min:
                    lo = max(1, b_min)
                    if cur_B < lo:
                        cur_B = lo
                    if cur_B > b_max:
                        cur_B = b_max
                    if mean == mean:
                        if mean > target_acc + 0.10 and cur_B < b_max:
                            cur_B += 1
                        elif mean < target_acc - 0.10 and cur_B > lo:
                            cur_B -= 1
            l = committed
    return out[len(prompt_bytes) :]


def generate_blockdiff(
    model, entropy_lm, patcher_config, prompt_bytes, new_bytes, config=None, stats=None
):
    if config is None:
        config = {}
    return generate_common(
        model, entropy_lm, patcher_config, prompt_bytes, new_bytes, config, False, stats
    )


def generate_blockdiff_verify(
    model, entropy_lm, patcher_config, prompt_bytes, new_bytes, config=None, stats=None
):
    if config is None:
        config = {}
    return generate_common(
        model, entropy_lm, patcher_config, prompt_bytes, new_bytes, config, True, stats
    )
