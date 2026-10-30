# ruff: noqa: E402
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VENV = os.path.join(_REPO, "pysrc", ".venv", "lib", "python3.11", "site-packages")
if os.path.isdir(_VENV):
    sys.path.insert(0, _VENV)
sys.path.insert(0, os.path.join(_REPO, "pysrc"))

import torch

from .checkpoint import load_c_checkpoint
from .models.blt import BLT
from .train import run_training


def _shape(cfg):
    layers = getattr(cfg, "layers", 0) or 2
    enc = cfg.enc_layers or layers
    glob = cfg.glob_layers or layers
    dec = cfg.dec_layers or layers
    return enc, glob, dec


def run_train(cfg, args=None):
    _ = run_training(cfg)
    return 0


def run_infer(cfg):
    enc, glob, dec = _shape(cfg)
    model = BLT(
        embed_dim=cfg.embed or 64,
        hidden_dim=cfg.hidden or 128,
        enc_layers=enc,
        glob_layers=glob,
        dec_layers=dec,
        max_seq_len=cfg.max_seq_len or 1024,
        d0_mode=cfg.d0_mode if hasattr(cfg, "d0_mode") else "learned",
    )
    if cfg.checkpoint:
        load_c_checkpoint(cfg.checkpoint, model)
    from .models.entropy_lm import EntropyLM

    entropy_lm = None
    if getattr(cfg, "entropy_lm", None):
        entropy_lm = EntropyLM(max_seq_len=cfg.max_seq_len or 1024)
        try:
            entropy_lm.load_state_dict(torch.load(cfg.entropy_lm, map_location="cpu"))
        except Exception:
            pass
    else:
        entropy_lm = EntropyLM(max_seq_len=cfg.max_seq_len or 1024)
    prompt = cfg.prompt or ""
    if cfg.prompt_file:
        with open(cfg.prompt_file) as f:
            prompt = f.read()
    prompt_bytes = [ord(c) % 256 for c in prompt]
    new_bytes = getattr(cfg, "new_bytes", 64)
    from .infer.block_dv import generate_blockdiff, generate_blockdiff_verify
    from .infer.generate import generate_greedy
    from .infer.self_spec import generate_greedy_selfspec
    from .patching import PatcherConfig

    pcfg = PatcherConfig(
        threshold_global=getattr(cfg, "patch_threshold_global", 2.5),
        threshold_monotonic=getattr(cfg, "patch_threshold_monotonic", 1.0),
        max_patch_length=getattr(cfg, "max_patch_length", 16),
        reset_on_newline=False,
    )
    method = getattr(cfg, "method", "greedy")
    if method == "greedy":
        out = generate_greedy(model, entropy_lm, pcfg, prompt_bytes, new_bytes)
    elif method == "blockdiff":
        out = generate_blockdiff(
            model,
            entropy_lm,
            pcfg,
            prompt_bytes,
            new_bytes,
            block_size=getattr(cfg, "block_size", 4),
        )
    elif method == "blockdv":
        cfgd = {"block_size": getattr(cfg, "block_size", 4)}
        out = generate_blockdiff_verify(
            model, entropy_lm, pcfg, prompt_bytes, new_bytes, config=cfgd
        )
    elif method == "selfspec":
        cfgs = {"window_k": getattr(cfg, "k", 8), "d0_mode": getattr(cfg, "d0_mode", "zeros")}
        out = generate_greedy_selfspec(model, entropy_lm, pcfg, prompt_bytes, new_bytes, cfgs)
    else:
        out = generate_greedy(model, entropy_lm, pcfg, prompt_bytes, new_bytes)
    text = "".join(chr(b) for b in out)
    if cfg.output:
        with open(cfg.output, "w") as f:
            f.write(text)
    else:
        print(text)
    return 0
