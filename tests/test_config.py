import dataclasses
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fblt._config import InferConfig, TrainConfig, load_config

PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)


# ------------------------------------------------------------------
# Test 1: TrainConfig round-trip
#
def test_train_roundtrip():
    print("[test 1] TrainConfig round-trip")
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

    check("--corpus present", "--corpus" in argv)
    check("corpus value", "data/train.bin" in argv)
    check("--steps present", "--steps" in argv)
    idx = argv.index("--steps")
    check("steps value", argv[idx + 1] == "100")
    check("--lr present", "--lr" in argv)
    idx = argv.index("--lr")
    check("lr value", argv[idx + 1] == "0.01")
    check("--embed present", "--embed" in argv)
    idx = argv.index("--embed")
    check("embed value", argv[idx + 1] == "128")
    check("--hidden present", "--hidden" in argv)
    idx = argv.index("--hidden")
    check("hidden value", argv[idx + 1] == "256")
    check("--layers present", "--layers" in argv)
    idx = argv.index("--layers")
    check("layers value", argv[idx + 1] == "3")

    # Bare switches: only emit when True
    check("--deterministic as bare switch", "--deterministic" in argv)
    check("--entropy-patches as bare switch", "--entropy-patches" in argv)

    # Diffusion is int 0, default is 1, should appear
    check("--diffusion present when non-default", "--diffusion" in argv)

    # Optional fields skipped when None
    default_cfg = TrainConfig(corpus="data/train.bin")
    default_argv = default_cfg.to_argv()
    check("default skips save-weights", "--save-weights" not in default_argv)
    check("default skips load-weights", "--load-weights" not in default_argv)
    check("default skips eval-corpus", "--eval-corpus" not in default_argv)
    check("default skips grad-norm-log", "--grad-norm-log" not in default_argv)
    check("default skips entropy-lm", "--entropy-lm" not in default_argv)
    check("default skips train-entropy-lm", "--train-entropy-lm" not in default_argv)

    # Optional list: lr_decay_steps
    cfg2 = TrainConfig(corpus="data/train.bin", lr_decay_steps=[2000, 4000])
    argv2 = cfg2.to_argv()
    check("lr_decay_steps present", "--lr-decay-steps" in argv2)
    idx = argv2.index("--lr-decay-steps")
    check("lr_decay_steps value", argv2[idx + 1] == "2000,4000")


# ------------------------------------------------------------------
# Test 2: InferConfig round-trip
#
def test_infer_roundtrip():
    print("[test 2] InferConfig round-trip")
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

    check("--checkpoint present", "--checkpoint" in argv)
    check("--embed present", "--embed" in argv)
    check("--hidden present", "--hidden" in argv)
    check("--enc-layers present", "--enc-layers" in argv)
    check("--glob-layers present", "--glob-layers" in argv)
    check("--dec-layers present", "--dec-layers" in argv)
    check("--backend present", "--backend" in argv)
    idx = argv.index("--backend")
    check("backend value", argv[idx + 1] == "cuda")
    check("--prompt present", "--prompt" in argv)
    idx = argv.index("--prompt")
    check("prompt value", argv[idx + 1] == "hello")
    check("--new-bytes present", "--new-bytes" in argv)
    idx = argv.index("--new-bytes")
    check("new-bytes value", argv[idx + 1] == "32")
    check("--method present", "--method" in argv)
    idx = argv.index("--method")
    check("method value", argv[idx + 1] == "blockdv")
    check("--block-size present", "--block-size" in argv)
    idx = argv.index("--block-size")
    check("block-size value", argv[idx + 1] == "16")
    check("--unmask present", "--unmask" in argv)
    idx = argv.index("--unmask")
    check("unmask value", argv[idx + 1] == "eb")
    check("--threshold present", "--threshold" in argv)
    idx = argv.index("--threshold")
    check("threshold value", argv[idx + 1] == "0.9")

    check("--boundary-aligned present", "--boundary-aligned" in argv)
    check("--adaptive present", "--adaptive" in argv)
    check("--fixed-patches present", "--fixed-patches" in argv)

    check("--cross-attn present", "--cross-attn" in argv)
    idx = argv.index("--cross-attn")
    check("cross-attn value", argv[idx + 1] == "last")
    check("--seed present", "--seed" in argv)
    idx = argv.index("--seed")
    check("seed value", argv[idx + 1] == "42")

    # skipped when none
    default_cfg = InferConfig(checkpoint="m.fblt", embed=1, hidden=1, enc_layers=1, glob_layers=1, dec_layers=1, backend="cpu")
    default_argv = default_cfg.to_argv()
    check("default skips prompt", "--prompt" not in default_argv)
    check("default skips prompt-file", "--prompt-file" not in default_argv)
    check("default skips entropy-lm", "--entropy-lm" not in default_argv)
    check("default skips output", "--output" not in default_argv)
    check("default skips threshold", "--threshold" not in default_argv)

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
    check(
        "threshold auto-adjust for eb",
        "--threshold" in argv_eb and argv_eb[argv_eb.index("--threshold") + 1] == "1.0",
        f"argv_eb={argv_eb}",
    )


# ------------------------------------------------------------------
# Test 3: YAML loading
#
def test_yaml_loading():
    print("[test 3] YAML loading")
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
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        cfg = load_config(TrainConfig, tmp_path)
        check("corpus loaded", cfg.corpus == "data/train.bin")
        check("steps loaded", cfg.steps == 500)
        check("lr loaded", cfg.lr == 0.02)
        check("embed loaded", cfg.embed == 96)
        check("hidden loaded", cfg.hidden == 192)
        check("layers loaded", cfg.layers == 4)
        check("diffusion loaded", cfg.diffusion == 0)
        check("seed loaded", cfg.seed == 42)
        check("block-size loaded", cfg.block_size == 8)
        check("t-min loaded", cfg.t_min == 0.05)
        check("optimizer loaded", cfg.optimizer == "adamw")
        check("beta1 loaded", cfg.beta1 == 0.95)
        check("beta2 loaded", cfg.beta2 == 0.98)
        check("eval-windows loaded", cfg.eval_windows == 100)
        check("report-every loaded", cfg.report_every == 10)
        check("deterministic loaded", cfg.deterministic is True)

        # Non-specified fields stay at defaults
        check("unspecified layers stays default 4", cfg.layers == 4)
        check("unspecified mask-scale stays default 1.0", cfg.mask_scale == 1.0)

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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f2:
            f2.write(infer_yaml)
            tmp_path2 = f2.name

        try:
            icfg = load_config(InferConfig, tmp_path2)
            check("infer checkpoint loaded", icfg.checkpoint == "runs/model.fblt")
            check("infer embed loaded", icfg.embed == 128)
            check("infer hidden loaded", icfg.hidden == 256)
            check("infer method loaded", icfg.method == "blockdv")
            check("infer new-bytes loaded", icfg.new_bytes == 32)
            check("infer seed loaded", icfg.seed == 99)
        finally:
            os.unlink(tmp_path2)
    finally:
        os.unlink(tmp_path)


# ------------------------------------------------------------------
# Test 4: Unknown YAML keys raise
#
def test_unknown_yaml_keys():
    print("[test 4] Unknown YAML keys raise")
    yaml_content = "bad-key: value\nanother-bad: 42\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        raised = False
        msg = ""
        try:
            load_config(TrainConfig, tmp_path)
        except ValueError as e:
            raised = True
            msg = str(e)
        check("raises ValueError", raised, f"no exception raised")
        check("error mentions bad key", "bad-key" in msg, f"msg={msg}")
    finally:
        os.unlink(tmp_path)


# ------------------------------------------------------------------
# Test 5: CLI override precedence
#
def test_override_precedence():
    print("[test 5] CLI override precedence")
    yaml_content = "corpus: data/train.bin\nlr: 0.01\nembed: 64\nsteps: 1000\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        cfg = load_config(TrainConfig, tmp_path, overrides=["lr=0.1"])
        check("yaml value not used", cfg.lr != 0.01)
        check("override wins", cfg.lr == 0.1)
        check("other fields unaffected", cfg.embed == 64)
        check("other fields unaffected steps", cfg.steps == 1000)

        # Multiple overrides
        cfg2 = load_config(
            TrainConfig, tmp_path, overrides=["lr=0.2", "embed=256"]
        )
        check("multi-override lr", cfg2.lr == 0.2)
        check("multi-override embed", cfg2.embed == 256)

        # Override with hyphenated key
        cfg3 = load_config(
            TrainConfig, tmp_path, overrides=["block-size=16"]
        )
        check("hyphenated override", cfg3.block_size == 16)

        # Unknown override raises
        raised = False
        try:
            load_config(TrainConfig, tmp_path, overrides=["bogus=123"])
        except ValueError:
            raised = True
        check("unknown override raises", raised)

        # Missing = raises
        raised2 = False
        try:
            load_config(TrainConfig, tmp_path, overrides=["badformat"])
        except ValueError:
            raised2 = True
        check("malformed override raises", raised2)
    finally:
        os.unlink(tmp_path)


# ------------------------------------------------------------------
# Test 6: C drift protection
#
def _extract_c_flags(path):
    with open(path, "r") as f:
        content = f.read()

    flags = set()
    for m in re.finditer(
        r'strcmp\s*\(\s*argv\s*\[\s*i\s*\]\s*,\s*"(--[a-z][a-z0-9-]*)"\s*\)',
        content,
    ):
        flag = m.group(1)
        if flag in ("--help",):
            continue
        flags.add(flag)
    return flags


def _flag_to_field(flag):
    return flag.lstrip("-").replace("-", "_")


def test_c_drift():
    print("[test 6] C drift protection")
    root = os.path.join(os.path.dirname(__file__), "..")

    train_c = os.path.join(root, "run", "train_blt_d.c")
    infer_c = os.path.join(root, "run", "infer.c")

    train_flags = _extract_c_flags(train_c)
    infer_flags = _extract_c_flags(infer_c)

    train_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    infer_fields = {f.name for f in dataclasses.fields(InferConfig)}

    missing_train = []
    for flag in sorted(train_flags):
        field = _flag_to_field(flag)
        if field not in train_fields:
            missing_train.append(f"{flag} → {field}")

    missing_infer = []
    for flag in sorted(infer_flags):
        field = _flag_to_field(flag)
        if field not in infer_fields:
            missing_infer.append(f"{flag} → {field}")

    check(
        "all train_blt_d.c flags in TrainConfig",
        len(missing_train) == 0,
        f"missing: {missing_train}" if missing_train else "",
    )
    check(
        "all infer.c flags in InferConfig",
        len(missing_infer) == 0,
        f"missing: {missing_infer}" if missing_infer else "",
    )

    # Report what was found
    print(f"  info  train_blt_d.c flags: {len(train_flags)}")
    print(f"  info  infer.c flags: {len(infer_flags)}")
    print(f"  info  TrainConfig fields: {len(train_fields)}")
    print(f"  info  InferConfig fields: {len(infer_fields)}")


# ------------------------------------------------------------------
# Main
#
if __name__ == "__main__":
    test_train_roundtrip()
    test_infer_roundtrip()
    test_yaml_loading()
    test_unknown_yaml_keys()
    test_override_precedence()
    test_c_drift()

    total = PASSED + FAILED
    print(f"\n{PASSED}/{total} tests passed")
    sys.exit(0 if FAILED == 0 else 1)
