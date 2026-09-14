import dataclasses

import pytest
import yaml

from fblt._config import InferConfig, TrainConfig, load_config
from fblt.infer import _SHAPE_FIELDS
from fblt.train import (
    _DEFAULT_OUTPUTS,
    _check_yaml_no_backend,
    _derive_run_name,
)


# ------------------------------------------------------------------
# Test 1: Run-directory creation and naming
#
def test_run_dir_naming(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # _derive_run_name with no config -> "default-{stamp}"
    name = _derive_run_name(None)
    assert name.startswith("default-")
    assert len(name) > len("default-")

    # _derive_run_name with config -> "{stem}-{stamp}"
    name2 = _derive_run_name("configs/train/debug.yaml")
    assert name2.startswith("debug-")
    assert len(name2) > len("debug-")

    # _derive_run_name with full path
    name3 = _derive_run_name("/some/path/production.yaml")
    assert name3.startswith("production-")

    # Actual directory creation
    run_name = _derive_run_name(None)
    run_dir = tmp_path / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    assert run_dir.is_dir()

    # Write a resolved_config.yaml
    cfg = TrainConfig(corpus="data/train.bin", embed=128)
    cfg.backend = "cpu"
    resolved_path = run_dir / "resolved_config.yaml"
    resolved_path.write_text(yaml.dump(dataclasses.asdict(cfg)))
    assert resolved_path.is_file()


# ------------------------------------------------------------------
# Test 2: resolved_config.yaml content
#
def test_resolved_config_content(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    run_dir = tmp_path / "runs" / "testrun"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        corpus="data/train.bin",
        embed=192,
        hidden=384,
        layers=2,
        steps=40000,
        lr=0.05,
    )
    cfg.backend = "cuda"
    cfg.save_weights = str(run_dir / "checkpoint.fblt")

    resolved_path = run_dir / "resolved_config.yaml"
    resolved_path.write_text(yaml.dump(dataclasses.asdict(cfg)))

    # Read it back
    data = yaml.safe_load(resolved_path.read_text())

    assert data.get("backend") == "cuda"
    assert data.get("embed") == 192
    assert data.get("hidden") == 384
    assert data.get("layers") == 2
    assert data.get("corpus") == "data/train.bin"
    assert data.get("steps") == 40000
    assert data.get("save_weights") == str(run_dir / "checkpoint.fblt")


# ------------------------------------------------------------------
# Test 3: Override precedence (load_config with yaml_path=None)
#
def test_override_precedence(tmp_path):
    # No YAML, just overrides
    cfg = load_config(TrainConfig, None, overrides=["lr=0.1", "embed=256"])
    assert cfg.lr == 0.1
    assert cfg.embed == 256
    assert cfg.steps == 2000

    # No YAML, no overrides -> pure defaults
    cfg2 = load_config(TrainConfig, None)
    assert cfg2.lr == 0.05
    assert cfg2.embed == 64

    # YAML + overrides
    cfg_file = tmp_path / "train.yaml"
    cfg_file.write_text("corpus: data/train.bin\nlr: 0.01\nembed: 64\n")

    cfg3 = load_config(TrainConfig, str(cfg_file), overrides=["lr=0.5"])
    assert cfg3.lr == 0.5
    assert cfg3.embed == 64


# ------------------------------------------------------------------
# Test 4: Backend-in-YAML rejection
#
def test_backend_rejection(tmp_path):
    # YAML with backend -> should raise SystemExit
    yaml_with_backend = tmp_path / "with_backend.yaml"
    yaml_with_backend.write_text("corpus: data/train.bin\nbackend: cuda\n")

    with pytest.raises(SystemExit):
        _check_yaml_no_backend(str(yaml_with_backend))

    # YAML without backend -> no error
    yaml_no_backend = tmp_path / "no_backend.yaml"
    yaml_no_backend.write_text("corpus: data/train.bin\nsteps: 100\n")
    _check_yaml_no_backend(str(yaml_no_backend))


# ------------------------------------------------------------------
# Test 5: Auto-shape matching
#
def test_auto_shape_matching(tmp_path):
    # Create a fake checkpoint dir with resolved_config.yaml
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    fake_ckpt = ckpt_dir / "model.fblt"
    fake_ckpt.write_text("fake")

    resolved_path = ckpt_dir / "resolved_config.yaml"
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
    resolved_path.write_text(yaml.dump(resolved_data))

    # InferConfig with defaults (embed=0, hidden=0) -> auto-detect fills them
    cfg = InferConfig()
    defaults = {f.name: f.default for f in dataclasses.fields(cfg)}
    updated = False
    for field_name in _SHAPE_FIELDS:
        current = getattr(cfg, field_name)
        default_val = defaults[field_name]
        if current == default_val and field_name in resolved_data:
            setattr(cfg, field_name, resolved_data[field_name])
            updated = True

    assert cfg.embed == 192
    assert cfg.hidden == 384
    assert cfg.enc_layers == 2
    assert cfg.glob_layers == 2
    assert cfg.dec_layers == 2
    assert cfg.cross_attn == "last"
    assert updated

    # set embed=128 so auto-detect doesn't override
    cfg2 = InferConfig(embed=128)
    defaults2 = {f.name: f.default for f in dataclasses.fields(cfg2)}
    for field_name in _SHAPE_FIELDS:
        current = getattr(cfg2, field_name)
        default_val = defaults2[field_name]
        if current == default_val and field_name in resolved_data:
            setattr(cfg2, field_name, resolved_data[field_name])

    assert cfg2.embed == 128
    assert cfg2.hidden == 384

    # No resolved_config.yaml so no changes
    empty_dir = tmp_path / "noresolved"
    empty_dir.mkdir()
    (empty_dir / "model.fblt").write_text("fake")

    cfg3 = InferConfig()
    resolved2 = empty_dir / "resolved_config.yaml"
    assert not resolved2.exists()
    assert cfg3.embed == 0

    # Empty resolved_config.yaml so no changes
    empty_resolved_dir = tmp_path / "emptyresolved"
    empty_resolved_dir.mkdir()
    (empty_resolved_dir / "model.fblt").write_text("fake")
    empty_resolved = empty_resolved_dir / "resolved_config.yaml"
    empty_resolved.write_text(yaml.dump({}))

    cfg4 = InferConfig()
    defaults4 = {f.name: f.default for f in dataclasses.fields(cfg4)}
    resolved4 = yaml.safe_load(empty_resolved.read_text()) or {}
    for field_name in _SHAPE_FIELDS:
        current = getattr(cfg4, field_name)
        default_val = defaults4[field_name]
        if current == default_val and field_name in resolved4:
            setattr(cfg4, field_name, resolved4[field_name])
    assert cfg4.embed == 0


# ------------------------------------------------------------------
# Test 6: Default output paths
#
def test_default_output_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    run_dir = tmp_path / "runs" / "myrun"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(corpus="data/train.bin")
    cfg.backend = "cpu"

    # Simulate the default path filling from train.py
    for field_name, filename in _DEFAULT_OUTPUTS.items():
        if getattr(cfg, field_name) is None:
            setattr(cfg, field_name, str(run_dir / filename))

    assert cfg.save_weights == str(run_dir / "checkpoint.fblt")
    assert cfg.grad_norm_log == str(run_dir / "grad_norm.log")
    assert cfg.loss_log == str(run_dir / "loss.log")
    assert cfg.batch_log == str(run_dir / "batch.log")

    # values should NOT be overridden
    cfg2 = TrainConfig(
        corpus="data/train.bin",
        save_weights="custom/path/model.fblt",
    )
    cfg2.backend = "cpu"
    for field_name, filename in _DEFAULT_OUTPUTS.items():
        if getattr(cfg2, field_name) is None:
            setattr(cfg2, field_name, str(run_dir / filename))

    assert cfg2.save_weights == "custom/path/model.fblt"
    assert cfg2.grad_norm_log == str(run_dir / "grad_norm.log")
