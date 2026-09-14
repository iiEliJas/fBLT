import argparse
import dataclasses
import os
import sys
import time
import typing

import yaml

from fblt._binary import resolve_binary
from fblt._config import TrainConfig, load_config
from fblt._runner import run_binary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fblt-train", description="Train BLT-D model"
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument(
        "--backend",
        choices=["cpu", "cuda"],
        required=True,
        help="Compute backend",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        help="key=value override (repeatable)",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Run name")
    return parser.parse_args()


def _check_yaml_no_backend(yaml_path: str) -> None:
    with open(yaml_path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    if isinstance(raw, dict) and "backend" in raw:
        raise SystemExit(
            f"Error: YAML config {yaml_path} contains 'backend' key. "
            "Use --backend on the command line instead."
        )


def _derive_run_name(config_path: typing.Optional[str]) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if config_path is not None:
        stem = os.path.splitext(os.path.basename(config_path))[0]
        return f"{stem}-{stamp}"
    return f"default-{stamp}"


_DEFAULT_OUTPUTS: typing.Dict[str, str] = {
    "save_weights": "checkpoint.fblt",
    "grad_norm_log": "grad_norm.log",
    "update_norm_log": "update_norm.log",
    "component_norm_log": "component_norm.log",
    "activation_dump_log": "activation_dump.log",
    "batch_log": "batch.log",
    "loss_log": "loss.log",
}


def main() -> None:
    args = _parse_args()

    # Reject backend in YAML
    if args.config is not None:
        _check_yaml_no_backend(args.config)

    # Build config
    cfg = load_config(TrainConfig, args.config, args.override)
    cfg.backend = args.backend

    # Derive run name and directory
    run_name = args.run_name if args.run_name is not None else _derive_run_name(args.config)
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Fill default output paths for any field still None
    for field_name, filename in _DEFAULT_OUTPUTS.items():
        if getattr(cfg, field_name) is None:
            setattr(cfg, field_name, os.path.join(run_dir, filename))

    # Write resolved config
    resolved_path = os.path.join(run_dir, "resolved_config.yaml")
    with open(resolved_path, "w") as fh:
        yaml.dump(dataclasses.asdict(cfg), fh)

    # Print run directory
    print(run_dir)

    # Resolve binary and run
    binary = resolve_binary("train_blt_d", cfg.backend)
    argv = cfg.to_argv()
    rc = run_binary(binary, argv)
    sys.exit(rc)
