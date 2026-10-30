import sys
import time

import torch

sys.path.insert(0, "pysrc")
from fblt_torch.decoder import LocalDecoder
from fblt_torch.encoder import LocalEncoder
from fblt_torch.global_model import GlobalTransformer
from fblt_torch.hash_ngram import HashNgram
from fblt_torch.ops import rope_precompute

E, H = 64, 128
cos, sin = rope_precompute(48, 16, 500000.0, "cpu", torch.float32)
ng = HashNgram([3, 4], 50, 1000000007, True, E)
enc = LocalEncoder(E, H, 2, 4, 4, cos, sin, ng, 0)
glob = GlobalTransformer(E, H, 2, 4, cos[:5], sin[:5])
dec = LocalDecoder(E, H, 2, 4, 4, cos, sin, 256, 0)

enc.eval()
glob.eval()
dec.eval()
x = torch.randint(0, 256, (8, 48), dtype=torch.uint8)
patches = [(0, 12), (12, 12), (24, 12), (36, 12)]
# adjust last
patches = [(0, 12), (12, 12), (24, 12), (36, 12)]

t0 = time.time()
with torch.no_grad():
    for _ in range(20):
        for b in range(8):
            P, h = enc(x[b], patches)
            g = glob(P)
            _ = dec(h, g, patches)
print("avg step ms:", (time.time() - t0) / 20 * 1000)
