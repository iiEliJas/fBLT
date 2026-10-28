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

    def _hash(self, bytes_seq: torch.Tensor, n: int):
        # returns list of (index, hash) for positions with valid hash
        N = bytes_seq.shape[0]
        if n > N:
            return []
        # rolling hash init
        prime = self.hash_prime
        mod = self.per_ngram_vocab
        # prime^n mod mod
        pow_val = 1 % mod
        for _ in range(n):
            pow_val = (pow_val * prime) % mod
        window = [0] * n
        ws = 0
        seen = 0
        cur = 0
        res = []
        for i in range(N):
            b = int(bytes_seq[i].item())
            outgoing = window[ws]
            window[ws] = b
            ws = (ws + 1) % n
            seen += 1
            if seen < n:
                continue
            if seen == n:
                h = 0
                idx = ws
                for _ in range(n):
                    h = (h * prime + window[idx]) % mod
                    idx = (idx + 1) % n
                cur = h
            else:
                updated = (cur * prime + b) % mod
                outgoing_contrib = (outgoing * pow_val) % mod
                updated = (updated + mod - outgoing_contrib) % mod
                cur = updated
            res.append((i, cur))
        return res

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
