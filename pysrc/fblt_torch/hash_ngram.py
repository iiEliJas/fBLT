import torch
import torch.nn as nn


class HashNgram(nn.Module):
    def __init__(self, ngram_sizes, per_ngram_vocab, hash_prime, normalize, embed_dim):
        super().__init__()
        self.ngram_sizes = list(ngram_sizes)
        self.per_ngram_vocab = per_ngram_vocab
        self.hash_prime = hash_prime
        self.normalize = normalize
        self.embed_dim = embed_dim
        self.tables = nn.ParameterList(
            [nn.Parameter(torch.empty(per_ngram_vocab, embed_dim)) for _ in self.ngram_sizes]
        )

    def _hash(self, bytes_seq: torch.Tensor, n: int) -> torch.Tensor:
        # bytes_seq: (N,) uint8 -> (N,) long hash ids
        N = bytes_seq.shape[0]
        out = torch.zeros(N, dtype=torch.long, device=bytes_seq.device)
        # rolling polynomial hash, simple O(N*n) reference (exact enough for parity)
        for i in range(N):
            h = 0
            for j in range(n):
                idx = i - (n - 1) + j
                if idx < 0:
                    b = 0
                else:
                    b = int(bytes_seq[idx].item())
                h = (h * 257 + b) % self.hash_prime
            out[i] = h % self.per_ngram_vocab
        return out

    def forward(self, bytes_seq: torch.Tensor) -> torch.Tensor:
        # bytes_seq: (N,) -> (N, embed_dim)
        N = bytes_seq.shape[0]
        acc = torch.zeros(N, self.embed_dim, device=bytes_seq.device, dtype=torch.float32)
        # byte embedding added by caller; here return ngram contribution only
        for ti, n in enumerate(self.ngram_sizes):
            ids = self._hash(bytes_seq, n)
            acc = acc + self.tables[ti][ids]
        if self.normalize:
            acc = acc / (len(self.ngram_sizes) + 1)
        return acc
