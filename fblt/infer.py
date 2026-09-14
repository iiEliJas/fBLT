import argparse
import dataclasses
import os
import sys
import typing

import yaml

from fblt._binary import resolve_binary
from fblt._config import InferConfig, load_config
from fblt._runner import run_binary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="fblt-infer", description="Run BLT inference")
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
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", type=str, default=None, help="Prompt text")
    prompt_group.add_argument("--prompt-file", type=str, default=None, help="Path to prompt file")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    return parser.parse_args()


def _check_yaml_no_backend(yaml_path: str) -> None:
    with open(yaml_path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    if isinstance(raw, dict) and "backend" in raw:
        raise SystemExit(
            f"Error: YAML config {yaml_path} contains 'backend' key. "
            "Use --backend on the command line instead."
        )


_SHAPE_FIELDS: typing.Tuple[str, ...] = (
    "embed",
    "hidden",
    "enc_layers",
    "glob_layers",
    "dec_layers",
    "cross_attn",
)


def main() -> None:
    args = _parse_args()

    # Reject backend in YAML
    if args.config is not None:
        _check_yaml_no_backend(args.config)

    # Build config
    cfg = load_config(InferConfig, args.config, args.override)

    # auto shape-matching from resolved_config.yaml next to checkpoint
    checkpoint_dir = os.path.dirname(args.checkpoint)
    resolved_path = os.path.join(checkpoint_dir, "resolved_config.yaml")
    if os.path.isfile(resolved_path):
        with open(resolved_path, "r") as fh:
            resolved = yaml.safe_load(fh) or {}
        if isinstance(resolved, dict):
            defaults = {f.name: f.default for f in dataclasses.fields(cfg)}
            for key in ("enc_layers", "glob_layers", "dec_layers"):
                if resolved.get(key, 0) == 0 and "layers" in resolved:
                    resolved[key] = resolved["layers"]
            updated = False
            for field_name in _SHAPE_FIELDS:
                current = getattr(cfg, field_name)
                default_val = defaults[field_name]
                if current == default_val and field_name in resolved:
                    setattr(cfg, field_name, resolved[field_name])
                    updated = True
            if updated:
                print("note: auto-detected shape from resolved_config.yaml", file=sys.stderr)

    # CLI overrides — these always win
    cfg.backend = args.backend
    cfg.checkpoint = args.checkpoint
    cfg.prompt = args.prompt
    cfg.prompt_file = args.prompt_file
    cfg.output = args.output

    # Resolve binary and run
    binary = resolve_binary("infer", cfg.backend)
    argv = cfg.to_argv()
    rc = run_binary(binary, argv)
    sys.exit(rc)
