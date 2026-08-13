import argparse
import struct
from pathlib import Path

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required to run this script. Install it with pip install torch.")

def write_tensor_fp32(path: Path, tensor: torch.Tensor) -> None:
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

def write_tensor_uint8(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.detach().cpu().contiguous().to(torch.uint8)
    shape = tuple(tensor.shape)
    values = tensor.view(-1).tolist()

    with path.open("wb") as fh:
        fh.write(struct.pack("<I", len(shape)))
        for dim in shape:
            fh.write(struct.pack("<I", dim))
        fh.write(bytes(values))

def build_patches(entropy: torch.Tensor, bytes_array: torch.Tensor, args):
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
        
        # Rule 1: Newline Context Reset
        if args.reset_on_newline and bytes_array[idx].item() == 0x0A:
            boundary = True
        # Rule 2: Max Length Cap
        elif current_len >= args.max_patch_length:
            boundary = True
        else:
            # Rule 3 & 4: Entropy Threshold Rules
            global_rule = val > args.threshold_global
            monotonic_rule = (idx > current_start) and ((val - float(entropy[idx-1].item())) > args.threshold_monotonic)
            
            if args.rule == 'global':
                boundary = global_rule
            elif args.rule == 'monotonic':
                boundary = monotonic_rule
            elif args.rule == 'both':
                boundary = global_rule or monotonic_rule

        # Save patch if boundary triggered
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

    # Flush final patch
    patch_starts.append(current_start)
    patch_lengths.append(current_len)
    patch_peak_entropies.append(current_peak)

    return patch_starts, patch_lengths, patch_peak_entropies


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump entropy reference data for Phase 1/2 tests.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Directory to write reference files")
    
    # Matching the exact config defaults found in the C tests
    parser.add_argument("--threshold-global", type=float, default=0.6, help="Global patch threshold")
    parser.add_argument("--threshold-monotonic", type=float, default=0.4, help="Monotonic patch threshold")
    parser.add_argument("--max-patch-length", type=int, default=5, help="Maximum length of a patch")
    parser.add_argument("--rule", type=str, choices=['global', 'monotonic', 'both'], default='both', help="Patching rule to apply")
    parser.add_argument("--reset-on-newline", action='store_true', default=True, help="Reset patches on newline bytes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    # 1. Generate probabilities and calculate entropy
    probs = torch.softmax(torch.randn(512, 256, dtype=torch.float32), dim=-1)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-9), dim=-1)
    
    # 2. Generate random bytes and inject newlines (0x0A) to test the structural reset rule
    bytes_array = torch.randint(0, 256, (512,), dtype=torch.uint8)
    newline_indices = torch.randint(1, 511, (15,))
    bytes_array[newline_indices] = 0x0A

    # 3. Simulate patching rules
    patch_starts, patch_lengths, patch_peak_entropies = build_patches(entropy, bytes_array, args)

    # 4. Export all binaries
    output_dir = args.output_dir
    write_tensor_fp32(output_dir / "phase1_probs.bin", probs)
    write_tensor_fp32(output_dir / "phase1_entropy.bin", entropy)
    write_tensor_uint8(output_dir / "phase1_bytes.bin", bytes_array)

    with (output_dir / "phase1_patch_boundaries.txt").open("w", encoding="utf-8") as fh:
        fh.write("# start length peak_entropy\n")
        for start, length, peak in zip(patch_starts, patch_lengths, patch_peak_entropies):
            fh.write(f"{start} {length} {peak:.6f}\n")

    print(f"Wrote {output_dir / 'phase1_probs.bin'}")
    print(f"Wrote {output_dir / 'phase1_entropy.bin'}")
    print(f"Wrote {output_dir / 'phase1_bytes.bin'}")
    print(f"Wrote {output_dir / 'phase1_patch_boundaries.txt'}")

if __name__ == "__main__":
    main()