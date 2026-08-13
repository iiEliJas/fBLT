import numpy as np
import argparse
import sys

def generate_entropy(size_mb):
    # 1 MB = 1024 * 1024 bytes. A 32-bit float is 4 bytes.
    num_elements = int((size_mb * 1024 * 1024) / 4)
    
    print(f"Generating {size_mb} MB of synthetic entropy data ({num_elements} FP32 values)...")
    
    data = np.random.uniform(low=0.0, high=4.0, size=num_elements).astype(np.float32)
    
    print(f"Writing binary data to data/dummy_entropy.bin ...")
    try:
        with open("data/dummy_entropy.bin", 'wb') as f:
            f.write(data.tobytes())
        print("Done! Ready for benchmarking.")
    except IOError as e:
        print(f"Error writing to file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate binary FP32 entropy data for BLT Patcher benchmark.")
    parser.add_argument("--size", type=float, default=100.0, help="Size in MB (default: 100)")
    
    args = parser.parse_args()
    generate_entropy(args.size)