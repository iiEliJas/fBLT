import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from golden import create_parser as create_parser_base
from golden import write_tensor_fp32, write_tensor_uint8


def build_patches(entropy, bytes_array, args):
    """Build patches based on entropy thresholding."""
    patch_starts = []
    patch_lengths = []
    patch_peak_entropies = []

    seq_len = entropy.shape[0]
    if seq_len == 0:
        return patch_starts, patch_lengths, patch_peak_entropies

    current_start = 0
    current_len = 1
    current_peak = float(entropy[0].item())

    for idx in range(1, seq_len):
        val = float(entropy[idx].item())
        boundary = False

        if args.reset_on_newline and bytes_array[idx].item() == 0x0A:
            boundary = True
        elif current_len >= args.max_patch_length:
            boundary = True
        else:
            global_rule = val > args.threshold_global
            monotonic_rule = (idx > current_start) and (
                (val - float(entropy[idx - 1].item())) > args.threshold_monotonic
            )

            if args.rule == "global":
                boundary = global_rule
            elif args.rule == "monotonic":
                boundary = monotonic_rule
            elif args.rule == "both":
                boundary = global_rule or monotonic_rule

        if boundary:
            patch_starts.append(current_start)
            patch_lengths.append(current_len)
            patch_peak_entropies.append(current_peak)
            current_start = idx
            current_len = 1
            current_peak = val
        else:
            current_len += 1
            if val > current_peak:
                current_peak = val

    patch_starts.append(current_start)
    patch_lengths.append(current_len)
    patch_peak_entropies.append(current_peak)

    return patch_starts, patch_lengths, patch_peak_entropies


def generate_golden_patcher_files(
    output_dir: str = "data/tests",
    threshold_global: float = 0.6,
    threshold_monotonic: float = 0.4,
    max_patch_length: int = 5,
    rule: str = "both",
    reset_on_newline: bool = True,
    seed: int = 0,
) -> None:
    """Generate golden files for patcher."""
    torch.manual_seed(seed)

    probs = torch.softmax(torch.randn(512, 256, dtype=torch.float32), dim=-1)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-9), dim=-1)

    bytes_array = torch.randint(0, 256, (512,), dtype=torch.uint8)
    newline_indices = torch.randint(1, 511, (15,))
    bytes_array[newline_indices] = 0x0A

    class Args:
        pass

    args = Args()
    args.threshold_global = threshold_global
    args.threshold_monotonic = threshold_monotonic
    args.max_patch_length = max_patch_length
    args.rule = rule
    args.reset_on_newline = reset_on_newline

    patch_starts, patch_lengths, patch_peak_entropies = build_patches(entropy, bytes_array, args)

    output_dir = Path(output_dir)
    write_tensor_fp32(output_dir / "patcher_probs.bin", probs)
    write_tensor_fp32(output_dir / "patcher_entropy.bin", entropy)
    write_tensor_uint8(output_dir / "patcher_bytes.bin", bytes_array)

    with (output_dir / "patcher_boundaries.txt").open("w", encoding="utf-8") as fh:
        fh.write("# start length peak_entropy\n")
        for start, length, peak in zip(patch_starts, patch_lengths, patch_peak_entropies):
            fh.write(f"{start} {length} {peak:.6f}\n")


def main_patcher():
    parser = create_parser_base("Generate golden files for patcher")
    parser.add_argument(
        "--threshold-global", type=float, default=0.6, help="Global patch threshold"
    )
    parser.add_argument(
        "--threshold-monotonic", type=float, default=0.4, help="Monotonic patch threshold"
    )
    parser.add_argument("--max-patch-length", type=int, default=5, help="Maximum length of a patch")
    parser.add_argument(
        "--rule",
        type=str,
        choices=["global", "monotonic", "both"],
        default="both",
        help="Patching rule",
    )
    parser.add_argument(
        "--reset-on-newline", action="store_true", default=True, help="Reset patches on newline"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    generate_golden_patcher_files(
        str(args.output_dir),
        threshold_global=args.threshold_global,
        threshold_monotonic=args.threshold_monotonic,
        max_patch_length=args.max_patch_length,
        rule=args.rule,
        reset_on_newline=args.reset_on_newline,
        seed=args.seed,
    )


if __name__ == "__main__":
    main_patcher()
