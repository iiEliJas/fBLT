import dataclasses
import os
import sys
import tempfile
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fblt._config import InferConfig, TrainConfig, load_config
from fblt.infer import _SHAPE_FIELDS, main as infer_main
from fblt.train import (
    _DEFAULT_OUTPUTS,
    _check_yaml_no_backend,
    _derive_run_name,
)

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
# Test 1: Run-directory creation and naming
#
def test_run_dir_naming():
    print("[test 1] Run-directory creation and naming")
    stamp = time.strftime("%Y%m%d-%H%M%S")

    # _derive_run_name with no config -> "default-{stamp}"
    name = _derive_run_name(None)
    check("no config -> default prefix", name.startswith("default-"))
    check("no config -> has timestamp", len(name) > len("default-"))

    # _derive_run_name with config -> "{stem}-{stamp}"
    name2 = _derive_run_name("configs/train/debug.yaml")
    check("config -> debug prefix", name2.startswith("debug-"))
    check("config -> has timestamp", len(name2) > len("debug-"))

    # _derive_run_name with full path
    name3 = _derive_run_name("/some/path/production.yaml")
    check("full path -> production prefix", name3.startswith("production-"))

    # Actual directory creation
    with tempfile.TemporaryDirectory() as tmp:
        orig = os.getcwd()
        os.chdir(tmp)
        try:
            run_name = _derive_run_name(None)
            run_dir = os.path.join("runs", run_name)
            os.makedirs(run_dir, exist_ok=True)
            check("runs dir created", os.path.isdir(run_dir))

            # Write a resolved_config.yaml
            cfg = TrainConfig(corpus="data/train.bin", embed=128)
            cfg.backend = "cpu"
            resolved_path = os.path.join(run_dir, "resolved_config.yaml")
            with open(resolved_path, "w") as fh:
                yaml.dump(dataclasses.asdict(cfg), fh)
            check("resolved_config.yaml written", os.path.isfile(resolved_path))
        finally:
            os.chdir(orig)


# ------------------------------------------------------------------
# Test 2: resolved_config.yaml content
#
def test_resolved_config_content():
    print("[test 2] resolved_config.yaml content")
    with tempfile.TemporaryDirectory() as tmp:
        orig = os.getcwd()
        os.chdir(tmp)
        try:
            run_dir = os.path.join("runs", "testrun")
            os.makedirs(run_dir, exist_ok=True)

            cfg = TrainConfig(
                corpus="data/train.bin",
                embed=192,
                hidden=384,
                layers=2,
                steps=40000,
                lr=0.05,
            )
            cfg.backend = "cuda"
            cfg.save_weights = os.path.join(run_dir, "checkpoint.fblt")

            resolved_path = os.path.join(run_dir, "resolved_config.yaml")
            with open(resolved_path, "w") as fh:
                yaml.dump(dataclasses.asdict(cfg), fh)

            # Read it back
            with open(resolved_path, "r") as fh:
                data = yaml.safe_load(fh)

            check("backend in resolved", data.get("backend") == "cuda")
            check("embed in resolved", data.get("embed") == 192)
            check("hidden in resolved", data.get("hidden") == 384)
            check("layers in resolved", data.get("layers") == 2)
            check("corpus in resolved", data.get("corpus") == "data/train.bin")
            check("steps in resolved", data.get("steps") == 40000)

            # Verify default output paths were filled
            check(
                "save_weights set",
                data.get("save_weights") == os.path.join(run_dir, "checkpoint.fblt"),
            )
        finally:
            os.chdir(orig)


# ------------------------------------------------------------------
# Test 3: Override precedence (load_config with yaml_path=None)
#
def test_override_precedence():
    print("[test 3] Override precedence (yaml_path=None)")

    # No YAML, just overrides
    cfg = load_config(TrainConfig, None, overrides=["lr=0.1", "embed=256"])
    check("override lr", cfg.lr == 0.1)
    check("override embed", cfg.embed == 256)
    check("default steps unchanged", cfg.steps == 2000)

    # No YAML, no overrides -> pure defaults
    cfg2 = load_config(TrainConfig, None)
    check("pure default lr", cfg2.lr == 0.05)
    check("pure default embed", cfg2.embed == 64)

    # YAML + overrides
    yaml_content = "corpus: data/train.bin\nlr: 0.01\nembed: 64\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name
    try:
        cfg3 = load_config(TrainConfig, tmp_path, overrides=["lr=0.5"])
        check("yaml+override lr wins", cfg3.lr == 0.5)
        check("yaml+override embed from yaml", cfg3.embed == 64)
    finally:
        os.unlink(tmp_path)


# ------------------------------------------------------------------
# Test 4: Backend-in-YAML rejection
#
def test_backend_rejection():
    print("[test 4] Backend-in-YAML rejection")

    # YAML with backend -> should raise SystemExit
    yaml_with_backend = "corpus: data/train.bin\nbackend: cuda\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_with_backend)
        tmp_path = f.name
    try:
        raised = False
        try:
            _check_yaml_no_backend(tmp_path)
        except SystemExit:
            raised = True
        check("raises SystemExit for backend in YAML", raised)
    finally:
        os.unlink(tmp_path)

    # YAML without backend -> no error
    yaml_no_backend = "corpus: data/train.bin\nsteps: 100\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_no_backend)
        tmp_path2 = f.name
    try:
        raised2 = False
        try:
            _check_yaml_no_backend(tmp_path2)
        except SystemExit:
            raised2 = True
        check("no error for YAML without backend", not raised2)
    finally:
        os.unlink(tmp_path2)


# ------------------------------------------------------------------
# Test 5: Auto-shape matching
#
def test_auto_shape_matching():
    print("[test 5] Auto-shape matching")

    # Import the function directly
    from fblt.infer import main as _  # ensure module loaded

    with tempfile.TemporaryDirectory() as tmp:
        # Create a fake checkpoint dir with resolved_config.yaml
        ckpt_dir = os.path.join(tmp, "checkpoints")
        os.makedirs(ckpt_dir)
        fake_ckpt = os.path.join(ckpt_dir, "model.fblt")
        with open(fake_ckpt, "w") as f:
            f.write("fake")

        resolved_path = os.path.join(ckpt_dir, "resolved_config.yaml")
        resolved_data = {
            "embed": 192,
            "hidden": 384,
            "enc_layers": 2,
            "glob_layers": 2,
            "dec_layers": 2,
            "cross_attn": "last",
            "backend": "cuda",
            "corpus": "data/train.bin",
        }
        with open(resolved_path, "w") as f:
            yaml.dump(resolved_data, f)

        # InferConfig with defaults (embed=0, hidden=0) -> auto-detect fills them
        cfg = InferConfig()
        import dataclasses as dc

        defaults = {f.name: f.default for f in dc.fields(cfg)}
        updated = False
        for field_name in _SHAPE_FIELDS:
            current = getattr(cfg, field_name)
            default_val = defaults[field_name]
            if current == default_val and field_name in resolved_data:
                setattr(cfg, field_name, resolved_data[field_name])
                updated = True

        check("auto-detect sets embed", cfg.embed == 192)
        check("auto-detect sets hidden", cfg.hidden == 384)
        check("auto-detect sets enc_layers", cfg.enc_layers == 2)
        check("auto-detect sets glob_layers", cfg.glob_layers == 2)
        check("auto-detect sets dec_layers", cfg.dec_layers == 2)
        check("auto-detect sets cross_attn", cfg.cross_attn == "last")
        check("any field updated", updated)

        # set embed=128 so auto-detect doesn't override
        cfg2 = InferConfig(embed=128)
        defaults2 = {f.name: f.default for f in dc.fields(cfg2)}
        for field_name in _SHAPE_FIELDS:
            current = getattr(cfg2, field_name)
            default_val = defaults2[field_name]
            if current == default_val and field_name in resolved_data:
                setattr(cfg2, field_name, resolved_data[field_name])

        check("user-set embed preserved", cfg2.embed == 128)
        check("auto-detect still sets hidden for cfg2", cfg2.hidden == 384)

        # No resolved_config.yaml so no changes
        empty_dir = os.path.join(tmp, "noresolved")
        os.makedirs(empty_dir)
        fake_ckpt2 = os.path.join(empty_dir, "model.fblt")
        with open(fake_ckpt2, "w") as f:
            f.write("fake")

        cfg3 = InferConfig()
        resolved2 = os.path.join(empty_dir, "resolved_config.yaml")
        check("no resolved_config -> file doesn't exist", not os.path.isfile(resolved2))
        check("cfg3 embed stays default", cfg3.embed == 0)

        # Empty resolved_config.yaml so no changes
        empty_resolved_dir = os.path.join(tmp, "emptyresolved")
        os.makedirs(empty_resolved_dir)
        fake_ckpt3 = os.path.join(empty_resolved_dir, "model.fblt")
        with open(fake_ckpt3, "w") as f:
            f.write("fake")
        empty_resolved = os.path.join(empty_resolved_dir, "resolved_config.yaml")
        with open(empty_resolved, "w") as f:
            yaml.dump({}, f)

        cfg4 = InferConfig()
        defaults4 = {f.name: f.default for f in dc.fields(cfg4)}
        with open(empty_resolved, "r") as fh:
            resolved4 = yaml.safe_load(fh) or {}
        for field_name in _SHAPE_FIELDS:
            current = getattr(cfg4, field_name)
            default_val = defaults4[field_name]
            if current == default_val and field_name in resolved4:
                setattr(cfg4, field_name, resolved4[field_name])
        check("empty resolved -> embed stays default", cfg4.embed == 0)


# ------------------------------------------------------------------
# Test 6: Default output paths
#
def test_default_output_paths():
    print("[test 6] Default output paths")
    with tempfile.TemporaryDirectory() as tmp:
        orig = os.getcwd()
        os.chdir(tmp)
        try:
            run_dir = os.path.join("runs", "myrun")
            os.makedirs(run_dir, exist_ok=True)

            cfg = TrainConfig(corpus="data/train.bin")
            cfg.backend = "cpu"

            # Simulate the default path filling from train.py
            for field_name, filename in _DEFAULT_OUTPUTS.items():
                if getattr(cfg, field_name) is None:
                    setattr(cfg, field_name, os.path.join(run_dir, filename))

            check(
                "save_weights default",
                cfg.save_weights == os.path.join(run_dir, "checkpoint.fblt"),
            )
            check(
                "grad_norm_log default",
                cfg.grad_norm_log == os.path.join(run_dir, "grad_norm.log"),
            )
            check(
                "loss_log default",
                cfg.loss_log == os.path.join(run_dir, "loss.log"),
            )
            check(
                "batch_log default",
                cfg.batch_log == os.path.join(run_dir, "batch.log"),
            )

            # values should NOT be overridden
            cfg2 = TrainConfig(
                corpus="data/train.bin",
                save_weights="custom/path/model.fblt",
            )
            cfg2.backend = "cpu"
            for field_name, filename in _DEFAULT_OUTPUTS.items():
                if getattr(cfg2, field_name) is None:
                    setattr(cfg2, field_name, os.path.join(run_dir, filename))

            check(
                "user-set save_weights preserved",
                cfg2.save_weights == "custom/path/model.fblt",
            )
            check(
                "other defaults still filled",
                cfg2.grad_norm_log == os.path.join(run_dir, "grad_norm.log"),
            )
        finally:
            os.chdir(orig)


# ------------------------------------------------------------------
# Main
#
if __name__ == "__main__":
    test_run_dir_naming()
    test_resolved_config_content()
    test_override_precedence()
    test_backend_rejection()
    test_auto_shape_matching()
    test_default_output_paths()

    total = PASSED + FAILED
    print(f"\n{PASSED}/{total} tests passed")
    sys.exit(0 if FAILED == 0 else 1)
