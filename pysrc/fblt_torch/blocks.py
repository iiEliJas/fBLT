import torch


def rng_next(state):
    state[0] = (state[0] + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state[0]
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return (z ^ (z >> 31)) & ((1 << 64) - 1)


def rng_uniform01(state):
    return (rng_next(state) >> 40) / (1 << 24)


def build_block_batch(bytes_seq, patches, block_size, rng_seed, t_in=0.0):
    N = len(bytes_seq)
    M = len(patches)
    NB = M - 1
    R = block_size * NB
    state = [rng_seed]
    t = t_in if t_in > 0.0 else rng_uniform01(state)
    if t <= 1e-6:
        t = 1e-6
    tokens = []
    positions = []
    targets = []
    cell_valid = []
    cell_masked = []
    groups = []
    for i in range(1, M):
        s = patches[i][0]
        for k in range(block_size):
            pos = s + k
            valid = pos < N
            positions.append(pos if valid else N - 1)
            targets.append(bytes_seq[pos] if valid else 0)
            masked_draw = (not valid) or (rng_uniform01(state) < t)
            tok = 256 if masked_draw else bytes_seq[pos]  # 256 = MASK
            tokens.append(tok)
            cell_valid.append(1 if valid else 0)
            cell_masked.append(1 if (valid and tok == 256) else 0)
            groups.append(i - 1)
    return {
        "num_clean": N,
        "block_size": block_size,
        "num_blocks": NB,
        "n_block_rows": R,
        "t": t,
        "tokens": torch.tensor(tokens, dtype=torch.long),
        "positions": torch.tensor(positions, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.uint8),
        "cell_valid": torch.tensor(cell_valid, dtype=torch.uint8),
        "cell_masked": torch.tensor(cell_masked, dtype=torch.uint8),
        "groups": torch.tensor(groups, dtype=torch.long),
    }


def figure5_mask(N, R, block_size, block_starts, num_blocks):
    S = N + R
    mask = torch.zeros(S, S)
    # clean rows causal
    for i in range(N):
        for j in range(N):
            if j <= i:
                mask[i, j] = 0.0
            else:
                mask[i, j] = float("-inf")
    # block rows: attend clean cutoff (block_start) + all block rows
    for bi in range(num_blocks):
        start = block_starts[bi]
        for k in range(block_size):
            row = N + bi * block_size + k
            for j in range(start):
                mask[row, j] = 0.0 if j < N else float("-inf")
            for j in range(N, S):
                mask[row, j] = 0.0
            for j in range(start, N):
                mask[row, j] = float("-inf")
    # clean rows do not attend block rows
    for i in range(N):
        for j in range(N, S):
            mask[i, j] = float("-inf")
    return mask
