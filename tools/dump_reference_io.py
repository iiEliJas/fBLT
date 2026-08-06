import math
import struct
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None


def _flatten_and_shape(value):
    if isinstance(value, (list, tuple)):
        if not value:
            return [], [0]

        flat = []
        child_flat, child_shape = _flatten_and_shape(value[0])
        shape = [len(value)] + child_shape
        for item in value:
            item_flat, _ = _flatten_and_shape(item)
            flat.extend(item_flat)
        return flat, shape

    return [float(value)], []


def _infer_shape(value):
    flat, shape = _flatten_and_shape(value)
    return shape, flat


def write_tensor(path: str, tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if torch is not None and hasattr(tensor, "tolist"):
        data = tensor.detach().cpu().contiguous().view(-1).to(torch.float32)
        shape = tuple(tensor.shape)
        values = data.tolist()
    else:
        shape, values = _infer_shape(tensor)

    with path.open("wb") as fh:
        ndim = len(shape)
        fh.write(struct.pack("<I", ndim))
        for dim in shape:
            fh.write(struct.pack("<I", int(dim)))
        for value in values:
            fh.write(struct.pack("<f", float(value)))


def _matmul(a, b):
    rows = len(a)
    inner = len(a[0])
    cols = len(b[0])
    out = []
    for row in range(rows):
        out_row = []
        for col in range(cols):
            total = 0.0
            for k in range(inner):
                total += a[row][k] * b[k][col]
            out_row.append(total)
        out.append(out_row)
    return out


def _softmax(values):
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps)
    return [x / total for x in exps]


def generate_golden_files(output_dir: str = "data") -> None:
    if torch is not None:
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
        out = torch.matmul(a, b)
        write_tensor(Path(output_dir) / "golden_matmul_a.bin", a)
        write_tensor(Path(output_dir) / "golden_matmul_b.bin", b)
        write_tensor(Path(output_dir) / "golden_matmul_out.bin", out)

        softmax_input = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=torch.float32)
        softmax_output = torch.softmax(softmax_input, dim=-1)
        write_tensor(Path(output_dir) / "golden_softmax_in.bin", softmax_input)
        write_tensor(Path(output_dir) / "golden_softmax_out.bin", softmax_output)
        return

    a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    b = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    out = _matmul(a, b)
    write_tensor(Path(output_dir) / "golden_matmul_a.bin", a)
    write_tensor(Path(output_dir) / "golden_matmul_b.bin", b)
    write_tensor(Path(output_dir) / "golden_matmul_out.bin", out)

    softmax_input = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
    softmax_output = [_softmax(row) for row in softmax_input]
    write_tensor(Path(output_dir) / "golden_softmax_in.bin", softmax_input)
    write_tensor(Path(output_dir) / "golden_softmax_out.bin", softmax_output)


if __name__ == "__main__":
    generate_golden_files()
