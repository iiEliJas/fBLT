import torch


def rng_next(state):
    state[0] = (state[0] + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state[0]
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return (z ^ (z >> 31)) & ((1 << 64) - 1)


def rng_uniform01_local(state):
    return (rng_next(state) >> 40) / (1 << 24)


def build_block_batch(bytes_seq, patches, block_size, rng_seed, t_in=0.0):
    N = len(bytes_seq)
    M = len(patches)
    NB = M - 1
    R = block_size * NB
    state = [rng_seed]
    t = t_in if t_in > 0.0 else rng_uniform01_local(state)
    if t <= 1e-6:
        t = 1e-6

    if NB <= 0 or R <= 0:
        return {
            "num_clean": N,
            "block_size": block_size,
            "num_blocks": 0,
            "n_block_rows": 0,
            "t": t,
            "tokens": torch.zeros(0, dtype=torch.long),
            "positions": torch.zeros(0, dtype=torch.long),
            "targets": torch.zeros(0, dtype=torch.uint8),
            "cell_valid": torch.zeros(0, dtype=torch.uint8),
            "cell_masked": torch.zeros(0, dtype=torch.uint8),
            "groups": torch.zeros(0, dtype=torch.long),
        }

    block_starts = torch.tensor([patches[i][0] for i in range(1, M)], dtype=torch.long)
    offsets = torch.arange(block_size, dtype=torch.long)
    positions = block_starts.repeat_interleave(block_size) + offsets.repeat(NB)
    valid = positions < N
    positions_clamped = positions.clamp(max=max(0, N - 1))
    byte_tensor = (
        torch.tensor(bytes_seq, dtype=torch.uint8) if N > 0 else torch.zeros(0, dtype=torch.uint8)
    )
    targets = torch.where(
        valid,
        byte_tensor[positions_clamped.clamp(max=max(0, N - 1))],
        torch.zeros_like(positions, dtype=torch.uint8),
    )
    # rng draws
    draws = torch.tensor([rng_uniform01_local(state) for _ in range(R)], dtype=torch.float32)
    masked = (~valid) | (draws < t)
    tokens = torch.where(masked, torch.full_like(positions, 256), targets.long())
    # bytes_seq lookup for unmasked
    vals = torch.where(
        valid,
        byte_tensor[positions_clamped.clamp(max=max(0, N - 1))],
        torch.zeros_like(positions, dtype=torch.uint8),
    )
    tokens = torch.where(masked, torch.full_like(positions, 256), vals.long())
    cell_valid = valid.to(torch.uint8)
    cell_masked = (valid & masked).to(torch.uint8)
    groups = torch.arange(NB, dtype=torch.long).repeat_interleave(block_size)
    return {
        "num_clean": N,
        "block_size": block_size,
        "num_blocks": NB,
        "n_block_rows": R,
        "t": t,
        "tokens": tokens.long(),
        "positions": positions_clamped.long(),
        "targets": targets.to(torch.uint8),
        "cell_valid": cell_valid,
        "cell_masked": cell_masked,
        "groups": groups.long(),
    }


def figure5_mask(N, R, block_size, block_starts, num_blocks):
    S = N + R
    mask = torch.full((S, S), float("-inf"))
    # clean rows causal within window
    # clean rows: j <= i
    ii = torch.arange(N)[:, None]
    jj = torch.arange(N)[None, :]
    mask[:N, :N] = torch.where(
        jj <= ii, torch.zeros_like(mask[:N, :N]), torch.full_like(mask[:N, :N], float("-inf"))
    )
    # block rows
    for bi in range(num_blocks):
        start = block_starts[bi]
        row0 = N + bi * block_size
        for k in range(block_size):
            row = row0 + k
            # attend clean bytes < start
            if start > 0:
                mask[row, : min(start, N)] = 0.0
            # attend all block rows
            mask[row, N:S] = 0.0
            # mask clean bytes >= start
            if start < N:
                mask[row, start:N] = float("-inf")
    # clean rows do not attend block rows
    mask[:N, N:S] = float("-inf")
    return mask
