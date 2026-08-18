from pathlib import Path
from golden import write_tensor_fp32, write_tensor_uint8, create_parser

def generate_data(output_dir="data", cross_attn_all_layers=True):
    out_path = Path(output_dir)
    suffix = "all_layers" if cross_attn_all_layers else "final_layer"

    bytes_in = torch.randint(0, 256, (24,), dtype=torch.uint8)
    write_tensor_uint8(out_path / f"local_encoder_bytes_in.bin", bytes_in)

    # reference forward through your PyTorch encoder reimplementation
    patch_out, byte_hidden_out = reference_local_encoder_forward(
        bytes_in, cross_attn_all_layers=cross_attn_all_layers
    )
    write_tensor_fp32(out_path / f"local_encoder_patch_out_{suffix}.bin", patch_out)
    write_tensor_fp32(out_path / f"local_encoder_byte_hidden_out_{suffix}.bin", byte_hidden_out)


if __name__ == "__main__":
    parser = create_parser("Generate local encoder golden files")
    args = parser.parse_args()
    generate_data(args.output_dir, cross_attn_all_layers=True)
    generate_data(args.output_dir, cross_attn_all_layers=False)