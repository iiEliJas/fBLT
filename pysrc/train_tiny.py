import sys

import torch

sys.path.insert(0, "pysrc")
from fblt_torch.decoder import LocalDecoder
from fblt_torch.losses import l_clean
from fblt_torch.ops import rope_precompute

cos, sin = rope_precompute(24, 8, 500000.0, "cpu", torch.float32)
m = LocalDecoder(32, 64, 2, 4, 4, cos, sin, 256, 8)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
bh = torch.randn(24, 32)
patch_in = torch.randn(5, 32)
patches = [(0, 4), (4, 5), (9, 4), (13, 5), (18, 6)]
bytes_t = torch.randint(0, 256, (24,), dtype=torch.uint8)
losses = []
for step in range(20):
    logits = m(bh, patch_in, patches)
    loss = l_clean(logits, bytes_t, clean_sum=False, has_window_plus_one=False)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
print("first", losses[0], "last", losses[-1])
assert losses[-1] < losses[0]
