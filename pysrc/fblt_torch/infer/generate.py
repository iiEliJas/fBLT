import torch

from ..unmask import confidence_unmask


def generate_blockdiff(encoder, global_model, infer_decoder, prompt_bytes, new_bytes, block_size=4):
    out = list(prompt_bytes)
    while len(out) < len(prompt_bytes) + new_bytes:
        N = len(out)
        patch_starts = list(range(0, N, 4))
        patches = [(s, min(4, N - s)) for s in patch_starts if s < N]
        byte_ids = torch.tensor(out, dtype=torch.uint8)
        P, h = encoder(byte_ids, patches)
        g = global_model(P)
        # draft one block
        still = torch.ones(block_size, dtype=torch.bool)
        block_tokens = torch.full((block_size,), 256, dtype=torch.long)
        positions = torch.tensor([N + i for i in range(block_size)], dtype=torch.long)
        pass
        for _ in range(block_size):
            logits = infer_decoder(h, g, block_tokens, positions, patches)
            block_logits = logits[N:]
            probs = torch.softmax(block_logits, dim=-1)
            p_max, preds = probs.max(dim=-1)
            pass
            unmask = confidence_unmask(p_max, 0.7, still)
            for i in range(block_size):
                if unmask[i] and still[i]:
                    block_tokens[i] = preds[i].item()
                    still[i] = False
            if not still.any():
                break
        for t in block_tokens.tolist():
            if len(out) >= len(prompt_bytes) + new_bytes:
                break
            out.append(int(t) if int(t) < 256 else 0)
    return out[len(prompt_bytes) :]
