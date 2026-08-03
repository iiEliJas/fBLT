import argparse
import struct
from pathlib import Path

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required to run this script. Install it with pip install torch.")


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.detach().cpu().contiguous().to(torch.float32)
    shape = tuple(tensor.shape)
    values = tensor.view(-1).tolist()

    with path.open("wb") as fh:
        fh.write(struct.pack("<I", len(shape)))
        for dim in shape:
            fh.write(struct.pack("<I", dim))
        for value in values:
            fh.write(struct.pack("<f", float(value)))


def build_patches(entropy: torch.Tensor, threshold: float):
    patch_starts = []
    patch_lengths = []
    patch_peak_entropies = []

    seq_len = entropy.shape[0]
    if seq_len == 0:
        return patch_starts, patch_lengths, patch_peak_entropies

    current_start = 0
    current_peak = float(entropy[0].item())

    for idx in range(1, seq_len):
        value = float(entropy[idx].item())
        if value >= threshold:
            patch_starts.append(current_start)
            patch_lengths.append(idx - current_start)
            patch_peak_entropies.append(current_peak)
            current_start = idx
            current_peak = value
        else:
            if value > current_peak:
                current_peak = value

    patch_starts.append(current_start)
    patch_lengths.append(seq_len - current_start)
    patch_peak_entropies.append(current_peak)

    return patch_starts, patch_lengths, patch_peak_entropies


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump entropy reference data for Phase 1 tests.")
    parser.add_argument("--output-dir", type=Path, default=Path("tests"), help="Directory to write reference files")
    parser.add_argument("--threshold", type=float, default=6.0, help="Patch threshold in bits")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    probs = torch.softmax(torch.randn(512, 256, dtype=torch.float32), dim=-1)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-9), dim=-1)

    patch_starts, patch_lengths, patch_peak_entropies = build_patches(entropy, args.threshold)

    output_dir = args.output_dir
    write_tensor(output_dir / "phase1_probs.bin", probs)
    write_tensor(output_dir / "phase1_entropy.bin", entropy)

    with (output_dir / "phase1_patch_boundaries.txt").open("w", encoding="utf-8") as fh:
        fh.write("# start length peak_entropy\n")
        for start, length, peak in zip(patch_starts, patch_lengths, patch_peak_entropies):
            fh.write(f"{start} {length} {peak:.6f}\n")

    print(f"Wrote {output_dir / 'phase1_probs.bin'}")
    print(f"Wrote {output_dir / 'phase1_entropy.bin'}")
    print(f"Wrote {output_dir / 'phase1_patch_boundaries.txt'}")


if __name__ == "__main__":
    main()
