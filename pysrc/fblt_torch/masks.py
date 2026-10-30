import torch


def figure5_mask(N, R, block_size, block_starts, num_blocks):
    S = N + R
    mask = torch.full((S, S), float("-inf"))
    for i in range(N):
        for j in range(N):
            if j <= i:
                mask[i, j] = 0.0
    for bi in range(num_blocks):
        start = block_starts[bi]
        for k in range(block_size):
            row = N + bi * block_size + k
            for j in range(start):
                if j < N:
                    mask[row, j] = 0.0
            for j in range(N, S):
                mask[row, j] = 0.0
    for i in range(N):
        for j in range(N, S):
            mask[i, j] = float("-inf")
    return mask


def cross_group_mask(q_groups, kv_groups):
    S = len(q_groups)
    M = len(kv_groups)
    mask = torch.full((S, M), float("-inf"))
    for i in range(S):
        gi = q_groups[i]
        if 0 <= gi < M:
            mask[i, gi] = 0.0
    return mask
