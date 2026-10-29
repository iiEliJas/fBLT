import torch


class KVCache:
    def __init__(self, decoder, max_seq_len):
        self.decoder = decoder
        self.max_seq_len = max_seq_len
        self.embed_dim = decoder.embed_dim
        self.k = 1
        self.patch_dim = self.embed_dim
        self.self_len = 0
        self.cross_num_patches = 0
        self.self_k = [torch.zeros(max_seq_len, self.embed_dim) for _ in decoder.layers]
        self.self_v = [torch.zeros(max_seq_len, self.embed_dim) for _ in decoder.layers]
        self.cross_k = [torch.zeros(max_seq_len, self.embed_dim) for _ in decoder.layers]
        self.cross_v = [torch.zeros(max_seq_len, self.embed_dim) for _ in decoder.layers]
        self.cached_patches = []

    def reset(self):
        self.self_len = 0
        self.cross_num_patches = 0
        self.cached_patches = []

    def truncate(self, self_len, num_patches):
        self.self_len = self_len
        self.cross_num_patches = num_patches
        if len(self.cached_patches) > num_patches:
            self.cached_patches = self.cached_patches[:num_patches]

    def common_patches(self, patches):
        common = 0
        while (
            common < self.cross_num_patches
            and common < len(patches)
            and self.cached_patches[common][0] == patches[common][0]
            and self.cached_patches[common][1] == patches[common][1]
        ):
            common += 1
        return common

    def refresh_cross(self, patch_in, patches, from_patch):
        if from_patch < len(self.cached_patches):
            self.cached_patches = self.cached_patches[:from_patch]
        for p in range(from_patch, len(patches)):
            self.cached_patches.append((patches[p][0], patches[p][1]))
        self.cross_num_patches = len(patches)

    def decode_step(self, patch_in, patches, d0_rows, trailing_closed=False):
        # Dense fallback: run decoder over cached prefix + new rows.
        n = d0_rows.shape[0]
        base = self.self_len
        x = torch.cat([torch.zeros(base, self.embed_dim), d0_rows], dim=0) if base > 0 else d0_rows
        # Fallback recomputes from d0 only.
        logits = self.decoder(x, patch_in, patches)
        self.self_len = base + n
        return logits[-n:]
