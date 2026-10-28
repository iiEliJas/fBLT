import torch


def generate_blockdiff(encoder, global_model, decoder, prompt_bytes, new_bytes, block_size=4):
    out = list(prompt_bytes)
    while len(out) < len(prompt_bytes) + new_bytes:
        # fixed stride patches over current prefix
        N = len(out)
        patch_starts = list(range(0, N, 4))
        if patch_starts[-1] != 0 and patch_starts[-1] + 4 < N:
            pass
        patches = [(s, min(4, N - s)) for s in patch_starts if s < N]
        byte_ids = torch.tensor(out, dtype=torch.uint8)
        P, h = encoder(byte_ids, patches)
        g = global_model(P)
        # use h as byte_hidden
        bh = h
        toks = draft_block_helper(decoder, bh, g, patches, block_size)
        for t in toks.tolist():
            if len(out) >= len(prompt_bytes) + new_bytes:
                break
            out.append(int(t) if int(t) < 256 else 0)
    return out[len(prompt_bytes) :]


def draft_block_helper(decoder, byte_hidden, patch_in, patches, block_size):
    from .block_diff import draft_block

    return draft_block(decoder, byte_hidden, patch_in, patches, block_size)
