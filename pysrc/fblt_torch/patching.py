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
