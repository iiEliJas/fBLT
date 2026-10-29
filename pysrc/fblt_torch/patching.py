def build_group_ids(patches, seq_len):
    # patches: list of (start,length)
    q = list(range(len(patches)))
    kv = [0] * seq_len
    pos = 0
    for j, (s, plen) in enumerate(patches):
        assert s == pos
        assert plen >= 1
        for i in range(plen):
            kv[pos + i] = j
        pos += plen
    assert pos == seq_len
    return q, kv


def decoder_latent_at(patches, pos):
    if pos < patches[0][0]:
        return 0
    for j, (start, plen) in enumerate(patches):
        if start > pos:
            break
        if pos < start + plen:
            if pos + 1 == start + plen:
                return j
            return 0 if j == 0 else j - 1
    return len(patches) - 1


def expand_group_ids(group_ids_in, k):
    out = []
    for gid in group_ids_in:
        out.extend([gid] * k)
    return out


class PatcherConfig:
    def __init__(
        self,
        threshold_global=2.5,
        threshold_monotonic=1.0,
        max_patch_length=16,
        rule="global",
        reset_on_newline=False,
    ):
        self.threshold_global = threshold_global
        self.threshold_monotonic = threshold_monotonic
        self.max_patch_length = max_patch_length
        self.rule = rule
        self.reset_on_newline = reset_on_newline


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def segment_patches(entropy_vals, bytes_seq, max_patches=1024, config=None):
    if config is None:
        config = PatcherConfig()
    tg = _cfg_get(config, "threshold_global", 2.5)
    tm = _cfg_get(config, "threshold_monotonic", 1.0)
    mpl = _cfg_get(config, "max_patch_length", 16)
    rule = _cfg_get(config, "rule", "global")
    reset = _cfg_get(config, "reset_on_newline", False)
    seq_len = len(bytes_seq)
    patches = []
    start = 0
    plen = 1
    peak = entropy_vals[0]
    for i in range(1, seq_len):
        boundary = False
        if reset and i >= 1 and bytes_seq[i - 1] == 10:
            boundary = True
        if plen >= mpl:
            boundary = True
        cur = entropy_vals[i - 1]
        prev = entropy_vals[i - 2] if i >= 2 else cur
        if rule in ("global", "both") and cur > tg:
            boundary = True
        if rule in ("monotonic", "both") and i >= start + 2 and (cur - prev) > tm:
            boundary = True
        if boundary:
            patches.append((start, plen))
            if len(patches) >= max_patches:
                break
            start = i
            plen = 1
            peak = entropy_vals[i]
        else:
            plen += 1
            if entropy_vals[i] > peak:
                peak = entropy_vals[i]
    if len(patches) < max_patches:
        patches.append((start, plen))
    return patches


def next_starts_patch(bytes_seq, pos, entropy_vals, patch_start, patch_len, config=None):
    if config is None:
        config = PatcherConfig()
    if pos == 0:
        return 0
    # boundary at pos
    i = pos
    if _cfg_get(config, "reset_on_newline", False) and i >= 1 and bytes_seq[i - 1] == 10:
        return 1
    if patch_len >= _cfg_get(config, "max_patch_length", 16):
        return 1
    cur = entropy_vals[i - 1]
    prev = entropy_vals[i - 2] if i >= 2 else cur
    tg = _cfg_get(config, "threshold_global", 2.5)
    tm = _cfg_get(config, "threshold_monotonic", 1.0)
    rule = _cfg_get(config, "rule", "global")
    if rule in ("global", "both") and cur > tg:
        return 1
    if rule in ("monotonic", "both") and i >= patch_start + 2 and (cur - prev) > tm:
        return 1
    return 0


def compute_entropy_vals(entropy_lm, byte_ids):
    import torch

    with torch.no_grad():
        logits = entropy_lm(torch.tensor(byte_ids, dtype=torch.uint8))
        probs = torch.softmax(logits, dim=-1)
        return (-(probs * torch.log(probs + 1e-12)).sum(dim=-1)).tolist()
