import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from golden import create_parser, write_tensor

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required to run this script. Install it with pip install torch.")


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


def generate_golden_math_files(output_dir: str = "data/tests") -> None:
    """Generate golden files for math operations (matmul, softmax)."""
    if torch is not None:
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
        out = torch.matmul(a, b)
        write_tensor(Path(output_dir) / "math_matmul_a.bin", a)
        write_tensor(Path(output_dir) / "math_matmul_b.bin", b)
        write_tensor(Path(output_dir) / "math_matmul_out.bin", out)

        softmax_input = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=torch.float32)
        softmax_output = torch.softmax(softmax_input, dim=-1)
        write_tensor(Path(output_dir) / "math_softmax_in.bin", softmax_input)
        write_tensor(Path(output_dir) / "math_softmax_out.bin", softmax_output)
        return

    a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    b = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    out = _matmul(a, b)
    write_tensor(Path(output_dir) / "math_matmul_a.bin", a)
    write_tensor(Path(output_dir) / "math_matmul_b.bin", b)
    write_tensor(Path(output_dir) / "math_matmul_out.bin", out)

    softmax_input = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
    softmax_output = [_softmax(row) for row in softmax_input]
    write_tensor(Path(output_dir) / "math_softmax_in.bin", softmax_input)
    write_tensor(Path(output_dir) / "math_softmax_out.bin", softmax_output)


def main_math():
    parser = create_parser("Generate golden files for math ops")
    args = parser.parse_args()
    generate_golden_math_files(str(args.output_dir))


if __name__ == "__main__":
    main_math()
