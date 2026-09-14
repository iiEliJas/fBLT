import argparse
import struct
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def write_tensor(path: str, tensor, dtype: str = "float32") -> None:
    """Write a tensor to binary file with shape header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Check explicitly if it is a PyTorch Tensor
    if torch is not None and isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().contiguous()
        if dtype == "float32":
            tensor = tensor.to(torch.float32)
            values = tensor.view(-1).tolist()
        elif dtype == "uint8":
            tensor = tensor.to(torch.uint8)
            values = tensor.view(-1).tolist()
        shape = tuple(tensor.shape)
    else:
        if isinstance(tensor, (list, tuple)):
            tensor = np.array(tensor)
        tensor = np.ascontiguousarray(tensor)
        if dtype == "float32":
            tensor = tensor.astype(np.float32)
        elif dtype == "uint8":
            tensor = tensor.astype(np.uint8)
        shape = tuple(tensor.shape)
        values = tensor.flatten().tolist()

    with path.open("wb") as fh:
        fh.write(struct.pack("<I", len(shape)))
        for dim in shape:
            fh.write(struct.pack("<I", int(dim)))

        if dtype == "uint8":
            fh.write(bytes(values))
        else:
            for value in values:
                fh.write(struct.pack("<f", float(value)))


def write_tensor_fp32(path: Path, tensor) -> None:
    """Write a float32 tensor."""
    write_tensor(str(path), tensor, dtype="float32")


def write_tensor_uint8(path: Path, tensor) -> None:
    """Write a uint8 tensor."""
    write_tensor(str(path), tensor, dtype="uint8")


def flatten_and_shape(value):
    """Flatten nested list/tuple and return shape."""
    if isinstance(value, (list, tuple)):
        if not value:
            return [], [0]
        flat = []
        child_flat, child_shape = flatten_and_shape(value[0])
        shape = [len(value)] + child_shape
        for item in value:
            item_flat, _ = flatten_and_shape(item)
            flat.extend(item_flat)
        return flat, shape
    return [float(value)], []


def infer_shape(value):
    """Infer shape from nested structure."""
    flat, shape = flatten_and_shape(value)
    return shape, flat


def get_output_dir(args) -> Path:
    """Get output directory from args or use default."""
    if hasattr(args, "output_dir"):
        return Path(args.output_dir)
    elif hasattr(args, "outdir"):
        return Path(args.outdir)
    else:
        return Path("data")


def create_parser(description: str = "Generate golden test data") -> argparse.ArgumentParser:
    """Create a basic argument parser."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tests"),
        help="Directory to write reference files",
    )
    return parser
