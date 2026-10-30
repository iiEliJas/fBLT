import json
import struct

import torch


def _read_tensor(fh):
    (name_len,) = struct.unpack("<H", fh.read(2))
    name = fh.read(name_len).decode("utf-8")
    (ndim,) = struct.unpack("<I", fh.read(4))
    dims = [struct.unpack("<Q" if False else "<I", fh.read(4))[0] for _ in range(ndim)]
    # dims are size_t in C; file uses 4 bytes? C writes sizeof(size_t) (8 on 64-bit). Handle 8.
    # Rewind: above assumed 4; redo properly
    return name, dims


def read_checkpoint(path):
    with open(path, "rb") as fh:
        magic = fh.read(4)
        assert magic == b"FBLT"
        (version,) = struct.unpack("<I", fh.read(4))
        meta = {}
        if version == 2:
            (meta_len,) = struct.unpack("<I", fh.read(4))
            meta = json.loads(fh.read(meta_len).decode("utf-8") or "{}")
        (num_tensors,) = struct.unpack("<I", fh.read(4))
        tensors = {}
        for _ in range(num_tensors):
            (name_len,) = struct.unpack("<H", fh.read(2))
            name = fh.read(name_len).decode("utf-8")
            (ndim,) = struct.unpack("<I", fh.read(4))
            dims = list(struct.unpack("<" + "Q" * ndim, fh.read(8 * ndim)))
            numel = 1
            for d in dims:
                numel *= d
            data = torch.frombuffer(fh.read(numel * 4), dtype=torch.float32).clone().reshape(dims)
            tensors[name] = data
        return version, meta, tensors


def write_checkpoint(path, version, meta, tensors):
    with open(path, "wb") as fh:
        fh.write(b"FBLT")
        fh.write(struct.pack("<I", version))
        if version == 2:
            meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
            fh.write(struct.pack("<I", len(meta_bytes)))
            fh.write(meta_bytes)
        fh.write(struct.pack("<I", len(tensors)))
        for name, t in tensors.items():
            nb = name.encode("utf-8")
            fh.write(struct.pack("<H", len(nb)))
            fh.write(nb)
            fh.write(struct.pack("<I", t.dim()))
            fh.write(struct.pack("<" + "Q" * t.dim(), *t.shape))
            fh.write(t.contiguous().view(-1).numpy().astype("float32").tobytes())


def _model_tensors(model):
    out = {}
    m = model
    # encoder
    out["enc.embed"] = m.encoder.byte_emb.weight.detach().cpu()
    for k, t in enumerate(m.encoder.ngram.tables):
        out[f"enc.ngram{k}"] = t.detach().cpu()
    for i, layer in enumerate(m.encoder.layers):
        out[f"enc.L{i}.norm1"] = layer.norm1.detach().cpu()
        out[f"enc.L{i}.attn_qkv"] = layer.attn.qkv.weight.detach().cpu().T.clone()
        out[f"enc.L{i}.attn_proj"] = layer.attn.proj.weight.detach().cpu().T.clone()
        out[f"enc.L{i}.norm2"] = layer.norm2.detach().cpu()
        out[f"enc.L{i}.ffn_up"] = layer.ffn_up.weight.detach().cpu().T.clone()
        out[f"enc.L{i}.ffn_gate"] = layer.ffn_gate.weight.detach().cpu().T.clone()
        out[f"enc.L{i}.ffn_down"] = layer.ffn_down.weight.detach().cpu().T.clone()
        out[f"enc.L{i}.cross_norm"] = m.encoder.cross_norms[i].detach().cpu()
        out[f"enc.L{i}.cross_q"] = m.encoder.cross_qs[i].weight.detach().cpu().T.clone()
        out[f"enc.L{i}.cross_k"] = m.encoder.cross_ks[i].weight.detach().cpu().T.clone()
        out[f"enc.L{i}.cross_v"] = m.encoder.cross_vs[i].weight.detach().cpu().T.clone()
        out[f"enc.L{i}.cross_proj"] = m.encoder.cross_projs[i].weight.detach().cpu().T.clone()
    # global
    for i, layer in enumerate(m.global_model.layers):
        out[f"glob.L{i}.norm1"] = layer.norm1.detach().cpu()
        out[f"glob.L{i}.attn_qkv"] = layer.attn.qkv.weight.detach().cpu().T.clone()
        out[f"glob.L{i}.attn_proj"] = layer.attn.proj.weight.detach().cpu().T.clone()
        out[f"glob.L{i}.norm2"] = layer.norm2.detach().cpu()
        out[f"glob.L{i}.ffn_up"] = layer.ffn_up.weight.detach().cpu().T.clone()
        out[f"glob.L{i}.ffn_gate"] = layer.ffn_gate.weight.detach().cpu().T.clone()
        out[f"glob.L{i}.ffn_down"] = layer.ffn_down.weight.detach().cpu().T.clone()
    # decoder
    for i, layer in enumerate(m.decoder.layers):
        out[f"dec.L{i}.norm1"] = layer.norm1.detach().cpu()
        out[f"dec.L{i}.attn_qkv"] = layer.attn.qkv.weight.detach().cpu().T.clone()
        out[f"dec.L{i}.attn_proj"] = layer.attn.proj.weight.detach().cpu().T.clone()
        out[f"dec.L{i}.norm2"] = layer.norm2.detach().cpu()
        out[f"dec.L{i}.ffn_up"] = layer.ffn_up.weight.detach().cpu().T.clone()
        out[f"dec.L{i}.ffn_gate"] = layer.ffn_gate.weight.detach().cpu().T.clone()
        out[f"dec.L{i}.ffn_down"] = layer.ffn_down.weight.detach().cpu().T.clone()
        out[f"dec.L{i}.cross_norm"] = m.decoder.cross_norms[i].detach().cpu()
        out[f"dec.L{i}.cross_q"] = m.decoder.cross_qs[i].weight.detach().cpu().T.clone()
        out[f"dec.L{i}.cross_k"] = m.decoder.cross_ks[i].weight.detach().cpu().T.clone()
        out[f"dec.L{i}.cross_v"] = m.decoder.cross_vs[i].weight.detach().cpu().T.clone()
        out[f"dec.L{i}.cross_proj"] = m.decoder.cross_projs[i].weight.detach().cpu().T.clone()
    out["dec.lm_head"] = m.decoder.lm_head.weight.detach().cpu().T.clone()
    if hasattr(m.decoder, "d0_embed"):
        out["dec.d0_embed"] = m.decoder.d0_embed.weight.detach().cpu()
    elif hasattr(m, "diffusion_decoder"):
        out["dec.d0_embed"] = m.diffusion_decoder.d0_embed.weight.detach().cpu()
    return out


def export_c_checkpoint(model, path, meta=None):
    tensors = _model_tensors(model)
    write_checkpoint(path, 2, meta or {}, tensors)


def load_c_checkpoint(path, model):
    _, meta, tensors = read_checkpoint(path)
    # inverse mapping
    m = model
    m.encoder.byte_emb.weight.data.copy_(tensors["enc.embed"])
    for k, t in enumerate(m.encoder.ngram.tables):
        t.data.copy_(tensors[f"enc.ngram{k}"])
    for i, layer in enumerate(m.encoder.layers):
        layer.norm1.data.copy_(tensors[f"enc.L{i}.norm1"])
        layer.attn.qkv.weight.data.copy_(tensors[f"enc.L{i}.attn_qkv"].T)
        layer.attn.proj.weight.data.copy_(tensors[f"enc.L{i}.attn_proj"].T)
        layer.norm2.data.copy_(tensors[f"enc.L{i}.norm2"])
        layer.ffn_up.weight.data.copy_(tensors[f"enc.L{i}.ffn_up"].T)
        layer.ffn_gate.weight.data.copy_(tensors[f"enc.L{i}.ffn_gate"].T)
        layer.ffn_down.weight.data.copy_(tensors[f"enc.L{i}.ffn_down"].T)
        m.encoder.cross_norms[i].data.copy_(tensors[f"enc.L{i}.cross_norm"])
        m.encoder.cross_qs[i].weight.data.copy_(tensors[f"enc.L{i}.cross_q"].T)
        m.encoder.cross_ks[i].weight.data.copy_(tensors[f"enc.L{i}.cross_k"].T)
        m.encoder.cross_vs[i].weight.data.copy_(tensors[f"enc.L{i}.cross_v"].T)
        m.encoder.cross_projs[i].weight.data.copy_(tensors[f"enc.L{i}.cross_proj"].T)
    for i, layer in enumerate(m.global_model.layers):
        layer.norm1.data.copy_(tensors[f"glob.L{i}.norm1"])
        layer.attn.qkv.weight.data.copy_(tensors[f"glob.L{i}.attn_qkv"].T)
        layer.attn.proj.weight.data.copy_(tensors[f"glob.L{i}.attn_proj"].T)
        layer.norm2.data.copy_(tensors[f"glob.L{i}.norm2"])
        layer.ffn_up.weight.data.copy_(tensors[f"glob.L{i}.ffn_up"].T)
        layer.ffn_gate.weight.data.copy_(tensors[f"glob.L{i}.ffn_gate"].T)
        layer.ffn_down.weight.data.copy_(tensors[f"glob.L{i}.ffn_down"].T)
    for i, layer in enumerate(m.decoder.layers):
        layer.norm1.data.copy_(tensors[f"dec.L{i}.norm1"])
        layer.attn.qkv.weight.data.copy_(tensors[f"dec.L{i}.attn_qkv"].T)
        layer.attn.proj.weight.data.copy_(tensors[f"dec.L{i}.attn_proj"].T)
        layer.norm2.data.copy_(tensors[f"dec.L{i}.norm2"])
        layer.ffn_up.weight.data.copy_(tensors[f"dec.L{i}.ffn_up"].T)
        layer.ffn_gate.weight.data.copy_(tensors[f"dec.L{i}.ffn_gate"].T)
        layer.ffn_down.weight.data.copy_(tensors[f"dec.L{i}.ffn_down"].T)
        m.decoder.cross_norms[i].data.copy_(tensors[f"dec.L{i}.cross_norm"])
        m.decoder.cross_qs[i].weight.data.copy_(tensors[f"dec.L{i}.cross_q"].T)
        m.decoder.cross_ks[i].weight.data.copy_(tensors[f"dec.L{i}.cross_k"].T)
        m.decoder.cross_vs[i].weight.data.copy_(tensors[f"dec.L{i}.cross_v"].T)
        m.decoder.cross_projs[i].weight.data.copy_(tensors[f"dec.L{i}.cross_proj"].T)
    m.decoder.lm_head.weight.data.copy_(tensors["dec.lm_head"].T)
    dd = getattr(m, "diffusion_decoder", None)
    if dd is not None and "dec.d0_embed" in tensors:
        dd.d0_embed.weight.data.copy_(tensors["dec.d0_embed"])
    if hasattr(m.decoder, "d0_embed"):
        m.decoder.d0_embed.weight.data.copy_(tensors["dec.d0_embed"])
    return meta
