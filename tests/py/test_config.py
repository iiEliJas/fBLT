import dataclasses
import os
import re

import pytest

from fblt._config import InferConfig, TrainConfig, load_config


# ------------------------------------------------------------------
# Test 1: TrainConfig round-trip
#
def test_train_roundtrip():
    cfg = TrainConfig(
        corpus="data/train.bin",
        steps=100,
        lr=0.01,
        embed=128,
        hidden=256,
        layers=3,
        diffusion=0,
        deterministic=True,
        entropy_patches=True,
        eval_corpus="data/eval.bin",
        save_weights="runs/w.fblt",
        grad_norm_log="logs/grad.log",
    )
    argv = cfg.to_argv()

    assert "--corpus" in argv
    assert "data/train.bin" in argv
    assert "--steps" in argv
    idx = argv.index("--steps")
    assert argv[idx + 1] == "100"
    assert "--lr" in argv
    idx = argv.index("--lr")
    assert argv[idx + 1] == "0.01"
    assert "--embed" in argv
    idx = argv.index("--embed")
    assert argv[idx + 1] == "128"
    assert "--hidden" in argv
    idx = argv.index("--hidden")
    assert argv[idx + 1] == "256"
    assert "--layers" in argv
    idx = argv.index("--layers")
    assert argv[idx + 1] == "3"

    # Bare switches: only emit when True
    assert "--deterministic" in argv
    assert "--entropy-patches" in argv

    # Diffusion is int 0, default is 1, should appear
    assert "--diffusion" in argv

    # Optional fields skipped when None
    default_cfg = TrainConfig(corpus="data/train.bin")
    default_argv = default_cfg.to_argv()
    assert "--save-weights" not in default_argv
    assert "--load-weights" not in default_argv
    assert "--eval-corpus" not in default_argv
    assert "--grad-norm-log" not in default_argv
    assert "--entropy-lm" not in default_argv
    assert "--train-entropy-lm" not in default_argv

    # Optional list: lr_decay_steps
    cfg2 = TrainConfig(corpus="data/train.bin", lr_decay_steps=[2000, 4000])
    argv2 = cfg2.to_argv()
    assert "--lr-decay-steps" in argv2
    idx = argv2.index("--lr-decay-steps")
    assert argv2[idx + 1] == "2000,4000"


# ------------------------------------------------------------------
# Test 2: InferConfig round-trip
#
def test_infer_roundtrip():
    cfg = InferConfig(
        checkpoint="runs/model.fblt",
        embed=128,
        hidden=256,
        enc_layers=2,
        glob_layers=2,
        dec_layers=2,
        backend="cuda",
        prompt="hello",
        new_bytes=32,
        method="blockdv",
        block_size=16,
        unmask="eb",
        threshold=0.9,
        boundary_aligned=True,
        adaptive=True,
        b_min=2,
        b_max=32,
        accept_target=0.7,
        adapt_window=16,
        cross_attn="last",
        fixed_patches=True,
        patch_threshold_global=3.0,
        patch_threshold_monotonic=0.5,
        max_patch_length=32,
        entropy_lm="runs/ent.fblt",
        output="out.bin",
        seed=42,
    )
    argv = cfg.to_argv()

    assert "--checkpoint" in argv
    assert "--embed" in argv
    assert "--hidden" in argv
    assert "--enc-layers" in argv
    assert "--glob-layers" in argv
    assert "--dec-layers" in argv
    assert "--backend" in argv
    idx = argv.index("--backend")
    assert argv[idx + 1] == "cuda"
    assert "--prompt" in argv
    idx = argv.index("--prompt")
    assert argv[idx + 1] == "hello"
    assert "--new-bytes" in argv
    idx = argv.index("--new-bytes")
    assert argv[idx + 1] == "32"
    assert "--method" in argv
    idx = argv.index("--method")
    assert argv[idx + 1] == "blockdv"
    assert "--block-size" in argv
    idx = argv.index("--block-size")
    assert argv[idx + 1] == "16"
    assert "--unmask" in argv
    idx = argv.index("--unmask")
    assert argv[idx + 1] == "eb"
    assert "--threshold" in argv
    idx = argv.index("--threshold")
    assert argv[idx + 1] == "0.9"

    assert "--boundary-aligned" in argv
    assert "--adaptive" in argv
    assert "--fixed-patches" in argv

    assert "--cross-attn" in argv
    idx = argv.index("--cross-attn")
    assert argv[idx + 1] == "last"
    assert "--seed" in argv
    idx = argv.index("--seed")
    assert argv[idx + 1] == "42"

    # skipped when none
    default_cfg = InferConfig(
        checkpoint="m.fblt",
        embed=1,
        hidden=1,
        enc_layers=1,
        glob_layers=1,
        dec_layers=1,
        backend="cpu",
    )
    default_argv = default_cfg.to_argv()
    assert "--prompt" not in default_argv
    assert "--prompt-file" not in default_argv
    assert "--entropy-lm" not in default_argv
    assert "--output" not in default_argv
    assert "--threshold" not in default_argv

    # Threshold auto-adjustment: unmask=eb, threshold=None, emit --threshold 1.0
    cfg_eb = InferConfig(
        checkpoint="m.fblt",
        embed=1,
        hidden=1,
        enc_layers=1,
        glob_layers=1,
        dec_layers=1,
        backend="cpu",
        unmask="eb",
        threshold=None,
    )
    argv_eb = cfg_eb.to_argv()
    assert "--threshold" in argv_eb
    assert argv_eb[argv_eb.index("--threshold") + 1] == "1.0"


# ------------------------------------------------------------------
# Test 3: YAML loading
#
def test_yaml_loading(tmp_path):
    yaml_content = (
        "corpus: data/train.bin\n"
        "steps: 500\n"
        "lr: 0.02\n"
        "embed: 96\n"
        "hidden: 192\n"
        "layers: 4\n"
        "diffusion: 0\n"
        "seed: 42\n"
        "block-size: 8\n"
        "t-min: 0.05\n"
        "optimizer: adamw\n"
        "beta1: 0.95\n"
        "beta2: 0.98\n"
        "eval-windows: 100\n"
        "report-every: 10\n"
        "deterministic: true\n"
    )
    cfg_file = tmp_path / "train.yaml"
    cfg_file.write_text(yaml_content)

    cfg = load_config(TrainConfig, str(cfg_file))
    assert cfg.corpus == "data/train.bin"
    assert cfg.steps == 500
    assert cfg.lr == 0.02
    assert cfg.embed == 96
    assert cfg.hidden == 192
    assert cfg.layers == 4
    assert cfg.diffusion == 0
    assert cfg.seed == 42
    assert cfg.block_size == 8
    assert cfg.t_min == 0.05
    assert cfg.optimizer == "adamw"
    assert cfg.beta1 == 0.95
    assert cfg.beta2 == 0.98
    assert cfg.eval_windows == 100
    assert cfg.report_every == 10
    assert cfg.deterministic is True

    # Non-specified fields stay at defaults
    assert cfg.mask_scale == 1.0

    # Also test InferConfig YAML loading
    infer_yaml = (
        "checkpoint: runs/model.fblt\n"
        "embed: 128\n"
        "hidden: 256\n"
        "enc-layers: 2\n"
        "glob-layers: 2\n"
        "dec-layers: 2\n"
        "backend: cuda\n"
        "method: blockdv\n"
        "new-bytes: 32\n"
        "block-size: 16\n"
        "seed: 99\n"
    )
    infer_file = tmp_path / "infer.yaml"
    infer_file.write_text(infer_yaml)

    icfg = load_config(InferConfig, str(infer_file))
    assert icfg.checkpoint == "runs/model.fblt"
    assert icfg.embed == 128
    assert icfg.hidden == 256
    assert icfg.method == "blockdv"
    assert icfg.new_bytes == 32
    assert icfg.seed == 99


# ------------------------------------------------------------------
# Test 4: Unknown YAML keys raise
#
def test_unknown_yaml_keys(tmp_path):
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text("bad-key: value\nanother-bad: 42\n")

    with pytest.raises(ValueError, match="bad-key"):
        load_config(TrainConfig, str(cfg_file))


# ------------------------------------------------------------------
# Test 5: CLI override precedence
#
def test_override_precedence(tmp_path):
    yaml_content = "corpus: data/train.bin\nlr: 0.01\nembed: 64\nsteps: 1000\n"
    cfg_file = tmp_path / "train.yaml"
    cfg_file.write_text(yaml_content)

    cfg = load_config(TrainConfig, str(cfg_file), overrides=["lr=0.1"])
    assert cfg.lr == 0.1
    assert cfg.embed == 64
    assert cfg.steps == 1000

    # Multiple overrides
    cfg2 = load_config(TrainConfig, str(cfg_file), overrides=["lr=0.2", "embed=256"])
    assert cfg2.lr == 0.2
    assert cfg2.embed == 256

    # Override with hyphenated key
    cfg3 = load_config(TrainConfig, str(cfg_file), overrides=["block-size=16"])
    assert cfg3.block_size == 16

    # Unknown override raises
    with pytest.raises(ValueError):
        load_config(TrainConfig, str(cfg_file), overrides=["bogus=123"])

    # Missing = raises
    with pytest.raises(ValueError):
        load_config(TrainConfig, str(cfg_file), overrides=["badformat"])


# ------------------------------------------------------------------
# Test 6: C drift protection
#
def _extract_c_flags(path):
    with open(path) as f:
        content = f.read()

    flags = set()
    for m in re.finditer(
        r'strcmp\s*\(\s*argv\s*\[\s*i\s*\]\s*,\s*"(--[a-z][a-z0-9-]*)"\s*\)',
        content,
    ):
        flag = m.group(1)
        if flag == "--help":
            continue
        flags.add(flag)
    return flags


def _flag_to_field(flag):
    return flag.lstrip("-").replace("-", "_")


def test_c_drift():
    root = os.path.join(os.path.dirname(__file__), "..", "..")

    train_c = os.path.join(root, "csrc", "train_blt_d.c")
    infer_c = os.path.join(root, "csrc", "infer.c")

    train_flags = _extract_c_flags(train_c)
    infer_flags = _extract_c_flags(infer_c)

    train_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    infer_fields = {f.name for f in dataclasses.fields(InferConfig)}

    missing_train = []
    for flag in sorted(train_flags):
        field = _flag_to_field(flag)
        if field not in train_fields:
            missing_train.append(f"{flag} -> {field}")

    missing_infer = []
    for flag in sorted(infer_flags):
        field = _flag_to_field(flag)
        if field not in infer_fields:
            missing_infer.append(f"{flag} -> {field}")

    assert not missing_train, f"train_blt_d.c flags missing from TrainConfig: {missing_train}"
    assert not missing_infer, f"infer.c flags missing from InferConfig: {missing_infer}"
