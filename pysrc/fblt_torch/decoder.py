import torch
import torch.nn as nn

from .masks import cross_group_mask, figure5_mask
from .ops import apply_rope, rmsnorm
from .transformer import TransformerLayer


def paper_decoder_latent_at(patches, pos):
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


class LocalDecoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_layers,
        num_heads,
        cross_heads,
        rope_cos,
        rope_sin,
        vocab_size=256,
        local_window=0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_heads = cross_heads
        self.head_dim = embed_dim // cross_heads
        self.layers = nn.ModuleList(
            [
                TransformerLayer(embed_dim, hidden_dim, num_heads, rope_cos, rope_sin)
                for _ in range(num_layers)
            ]
        )
        self.cross_norms = nn.ParameterList(
            [nn.Parameter(torch.ones(embed_dim)) for _ in range(num_layers)]
        )
        self.cross_qs = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)]
        )
        self.cross_ks = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)]
        )
        self.cross_vs = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)]
        )
        self.cross_projs = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.local_window = local_window

    def forward(self, byte_hidden, patch_in, patches, trailing_closed=False):
        P = patch_in
        if P.shape[1] != self.embed_dim:
            # k-split: mean over sub-tokens
            P = P.view(P.shape[0], -1, self.embed_dim).mean(1)
        x = byte_hidden.clone()
        for li, layer in enumerate(self.layers):
            # cross-attn: bytes query patches
            xn = rmsnorm(x, self.cross_norms[li])
            # group id per byte
            M = P.shape[0]
            Q = (
                (xn @ self.cross_qs[li].weight.T)
                .view(x.shape[0], self.cross_heads, self.head_dim)
                .transpose(0, 1)
            )
            Kmat = (
                (P @ self.cross_ks[li].weight.T)
                .view(M, self.cross_heads, self.head_dim)
                .transpose(0, 1)
            )
            Vmat = (
                (P @ self.cross_vs[li].weight.T)
                .view(M, self.cross_heads, self.head_dim)
                .transpose(0, 1)
            )
            # build mask: byte i attends latent paper_decoder_latent_at(patches,i)
            mask = torch.full((x.shape[0], M), float("-inf"))
            for i in range(x.shape[0]):
                gi = paper_decoder_latent_at(patches, i)
                if trailing_closed and i == x.shape[0] - 1:
                    gi = len(patches) - 1
                mask[i, gi] = 0.0
            scores = (Q @ Kmat.transpose(-2, -1)) / (self.head_dim**0.5)
            # mask shape (N,M) -> (1,N? need broadcast to (heads,N,M))
            scores = scores + mask.unsqueeze(0).unsqueeze(0) if False else scores
            scores = scores + mask.unsqueeze(0)
            probs = torch.softmax(scores, dim=-1)
            out = (probs @ Vmat).transpose(0, 1).reshape(x.shape[0], -1)
            x = x + self.cross_projs[li](out)
            x = layer(x, window=self.local_window)
        logits = self.lm_head(x)
        return logits


class DiffusionLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, cross_heads, head_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.cross_heads = cross_heads
        self.head_dim = head_dim
        self.norm1 = nn.Parameter(torch.ones(embed_dim))
        self.self_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.self_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm2 = nn.Parameter(torch.ones(embed_dim))
        self.ffn_up = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.ffn_gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.ffn_down = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.cross_norm = nn.Parameter(torch.ones(embed_dim))
        self.cross_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.cross_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.cross_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.cross_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        x,
        patch_in,
        patches,
        S,
        M,
        E,
        q_groups,
        kv_groups,
        self_mask,
        cos,
        sin,
        cross_heads,
        head_dim,
    ):
        # cross
        xn = rmsnorm(x, self.cross_norm)
        Q = (xn @ self.cross_q.weight.T).view(S, cross_heads, head_dim).transpose(0, 1)
        K = (patch_in @ self.cross_k.weight.T).view(M, cross_heads, head_dim).transpose(0, 1)
        V = (patch_in @ self.cross_v.weight.T).view(M, cross_heads, head_dim).transpose(0, 1)
        cmask = cross_group_mask(q_groups, kv_groups).unsqueeze(0)
        scores = (Q @ K.transpose(-2, -1)) / (head_dim**0.5) + cmask
        probs = torch.softmax(scores, dim=-1)
        out = (probs @ V).transpose(0, 1).reshape(S, -1)
        x = x + self.cross_proj(out)
        # self
        normed = rmsnorm(x, self.norm1)
        qkv = self.self_qkv(normed).view(S, 3, cross_heads, head_dim)
        q = qkv[:, 0].transpose(0, 1)
        kmat = qkv[:, 1].transpose(0, 1)
        vmat = qkv[:, 2].transpose(0, 1)
        q = q.permute(1, 0, 2).contiguous().view(S, cross_heads, head_dim)
        q = apply_rope(q, cos, sin)
        q = q.permute(1, 0, 2).contiguous().view(cross_heads, S, head_dim)
        kmat = kmat.permute(1, 0, 2).contiguous().view(S, cross_heads, head_dim)
        kmat = apply_rope(kmat, cos, sin)
        kmat = kmat.permute(1, 0, 2).contiguous().view(cross_heads, S, head_dim)
        scores2 = (q @ kmat.transpose(-2, -1)) / (head_dim**0.5) + self_mask.unsqueeze(0)
        probs2 = torch.softmax(scores2, dim=-1)
        attn_out = (probs2 @ vmat).transpose(0, 1).reshape(S, -1)
        attn_out = self.self_proj(attn_out)
        x = x + attn_out
        # ffn
        normed2 = rmsnorm(x, self.norm2)
        gate = self.ffn_gate(normed2)
        up = self.ffn_up(normed2)
        x = x + self.ffn_down(torch.nn.functional.silu(gate) * up)
        return x


class DiffusionDecoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_layers,
        num_heads,
        cross_heads,
        rope_cos,
        rope_sin,
        vocab_size=256,
        max_seq_len=4096,
        d0_mode="learned",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.cross_heads = cross_heads
        self.head_dim = embed_dim // cross_heads
        self.rope_cos = rope_cos
        self.rope_sin = rope_sin
        self.max_seq_len = max_seq_len
        self.d0_mode = d0_mode
        self.layers = nn.ModuleList(
            [
                DiffusionLayer(embed_dim, hidden_dim, num_heads, cross_heads, self.head_dim)
                for _ in range(num_layers)
            ]
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.d0_embed = nn.Embedding(257, embed_dim)

    def forward(
        self,
        byte_hidden,
        patch_in,
        patches,
        batch,
        clean_bytes=None,
        targets=None,
        trailing_closed=False,
    ):
        N = byte_hidden.shape[0]
        R = batch["n_block_rows"] if isinstance(batch, dict) else batch.n_block_rows
        S = N + R
        M = patch_in.shape[0]
        E = self.embed_dim
        x = torch.zeros(S, E, device=byte_hidden.device, dtype=byte_hidden.dtype)
        x[:N] = byte_hidden
        tokens = batch["tokens"] if isinstance(batch, dict) else batch.tokens
        if self.d0_mode == "learned":
            x[N:] = self.d0_embed(tokens.clamp(0, 256))
        positions = list(range(N)) + [
            int(p)
            for p in (batch["positions"] if isinstance(batch, dict) else batch.positions).tolist()
        ]
        cos = self.rope_cos[positions]
        sin = self.rope_sin[positions]
        q_groups = [0] * (N + R)
        for i in range(N):
            pos = i
            if pos < patches[0][0]:
                q_groups[i] = 0
                continue
            gi = 0
            for j, (start, plen) in enumerate(patches):
                if start > pos:
                    break
                if pos < start + plen:
                    if trailing_closed and i == N - 1:
                        gi = M - 1
                    elif pos + 1 == start + plen and j + 1 < M:
                        gi = j
                    else:
                        gi = 0 if j == 0 else j - 1
                    break
            else:
                gi = M - 1
            q_groups[i] = gi
        groups = batch["groups"] if isinstance(batch, dict) else batch.groups
        for r in range(R):
            q_groups[N + r] = int(groups[r].item() if torch.is_tensor(groups[r]) else groups[r])
        if trailing_closed and N > 0:
            q_groups[N - 1] = M - 1
        block_starts = [
            patches[j + 1][0]
            for j in range((batch["num_blocks"] if isinstance(batch, dict) else batch.num_blocks))
        ]
        block_size = batch["block_size"] if isinstance(batch, dict) else batch.block_size
        num_blocks = batch["num_blocks"] if isinstance(batch, dict) else batch.num_blocks
        self_mask = figure5_mask(N, R, block_size, block_starts, num_blocks)
        kv_groups = []
        for pi in range(M):
            kv_groups.extend([pi])
        for layer in self.layers:
            x = layer(
                x,
                patch_in,
                patches,
                S,
                M,
                E,
                q_groups,
                kv_groups,
                self_mask,
                cos,
                sin,
                self.cross_heads,
                self.head_dim,
            )
        logits = self.lm_head(x)
        loss = torch.tensor(0.0, device=x.device)
        cell_masked = batch["cell_masked"] if isinstance(batch, dict) else batch.cell_masked
        targets_b = batch["targets"] if isinstance(batch, dict) else batch.targets
        tval = batch["t"] if isinstance(batch, dict) else batch.t
        clean_sum = (
            batch.get("clean_sum", False)
            if isinstance(batch, dict)
            else getattr(batch, "clean_sum", False)
        )
        mask_norm = (
            batch.get("mask_norm", False)
            if isinstance(batch, dict)
            else getattr(batch, "mask_norm", False)
        )
        loss_scale = (
            batch.get("loss_scale", 1.0)
            if isinstance(batch, dict)
            else getattr(batch, "loss_scale", 1.0)
        )
        last_row_scale = (
            batch.get("last_row_scale", 1.0)
            if isinstance(batch, dict)
            else getattr(batch, "last_row_scale", 1.0)
        )
        clean_rows = N if targets is not None else max(N - 1, 0)
        if clean_rows > 0 and N >= 2:
            tgt = (
                targets[1 : 1 + clean_rows]
                if targets is not None
                else torch.tensor(
                    clean_bytes[1 : 1 + clean_rows], device=x.device, dtype=torch.long
                )
            )
            if targets is None:
                cb = torch.tensor(clean_bytes, device=x.device, dtype=torch.long)
                tgt = cb[1 : 1 + clean_rows]
            ce = torch.nn.functional.cross_entropy(
                logits[:clean_rows], tgt.long(), reduction="mean"
            )
            if clean_sum:
                ce = ce * clean_rows
            loss = loss + ce
            if targets is not None and last_row_scale != 1.0:
                last_ce = torch.nn.functional.cross_entropy(
                    logits[N - 1 : N], targets[N : N + 1].long(), reduction="mean"
                )
                loss = loss + (last_row_scale - 1.0) * last_ce / N
        if tval and tval > 0:
            l_mask = 0.0
            n_masked = 0
            for r in range(R):
                if (
                    cell_masked[r]
                    if not torch.is_tensor(cell_masked)
                    else bool(
                        cell_masked[r].item() if torch.is_tensor(cell_masked[r]) else cell_masked[r]
                    )
                ):
                    n_masked += 1
                    row = logits[N + r]
                    tgt = (
                        int(targets_b[r].item())
                        if torch.is_tensor(targets_b)
                        else int(targets_b[r])
                    )
                    l_mask += torch.nn.functional.cross_entropy(
                        row.unsqueeze(0), torch.tensor([tgt], device=row.device), reduction="sum"
                    ).item()
            if mask_norm and n_masked > 0:
                l_mask = l_mask / n_masked
            loss = loss + loss_scale * (l_mask / tval)
        return logits, loss


BlockDiffusionDecoder = DiffusionDecoder
