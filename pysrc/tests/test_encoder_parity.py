import struct
import sys

import numpy as np
import torch

sys.path.insert(0, "pysrc")
from fblt_torch.encoder import LocalEncoder
from fblt_torch.hash_ngram import HashNgram
from fblt_torch.ops import rope_precompute


def load(path):
    with open(path, "rb") as f:
        ndim = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<I", f.read(4))[0] for _ in range(ndim)]
        arr = np.fromfile(f, dtype="<f4").reshape(dims)
    return torch.from_numpy(arr.copy())


def load_u8(path):
    with open(path, "rb") as f:
        ndim = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<I", f.read(4))[0] for _ in range(ndim)]
        arr = np.fromfile(f, dtype=np.uint8).reshape(dims)
    return torch.from_numpy(arr.copy())


def test_encoder_parity():
    W = "data/tests/local_encoder_weights_all_layers"
    E, H, L = 32, 64, 3
    cos, sin = rope_precompute(24, 8, 500000.0, "cpu", torch.float32)
    ng = HashNgram([3, 4, 5], 512, 1000000007, True, E)
    for i in range(3):
        ng.tables[i].data = load(f"{W}/ngram_table_{i}.bin")
    m = LocalEncoder(E, H, L, 4, 4, cos, sin, ng, 8)
    m.byte_emb.weight.data = load(f"{W}/byte_embedding_weight.bin")
    for li in range(L):
        layer = m.layers[li]
        layer.norm1.data = load(f"{W}/layer_{li}_norm1_weight.bin")
        layer.attn.qkv.weight.data = load(f"{W}/layer_{li}_attn_qkv_w.bin").T.clone()
        layer.attn.proj.weight.data = load(f"{W}/layer_{li}_attn_proj_w.bin").T.clone()
        layer.norm2.data = load(f"{W}/layer_{li}_norm2_weight.bin")
        layer.ffn_up.weight.data = load(f"{W}/layer_{li}_ffn_up_w.bin").T.clone()
        gate_path = f"{W}/layer_{li}_ffn_gate_w.bin"
        layer.ffn_gate.weight.data = load(gate_path).T.clone()
        layer.ffn_down.weight.data = load(f"{W}/layer_{li}_ffn_down_w.bin").T.clone()
        m.cross_norms[li].data = load(f"{W}/layer_{li}_cross_norm_weight.bin")
        m.cross_qs[li].weight.data = load(f"{W}/layer_{li}_cross_weight_q.bin").T.clone()
        m.cross_ks[li].weight.data = load(f"{W}/layer_{li}_cross_weight_k.bin").T.clone()
        m.cross_vs[li].weight.data = load(f"{W}/layer_{li}_cross_weight_v.bin").T.clone()
        m.cross_projs[li].weight.data = load(f"{W}/layer_{li}_cross_weight_proj.bin").T.clone()
    bytes_t = load_u8("data/tests/local_encoder_bytes_in.bin")
    patches = [(0, 4), (4, 5), (9, 4), (13, 5), (18, 6)]
    with torch.no_grad():
        P, h = m(bytes_t, patches)
    exp = load("data/tests/local_encoder_byte_hidden_out_all_layers.bin")
    assert (h - exp).abs().max().item() < 1e-4
