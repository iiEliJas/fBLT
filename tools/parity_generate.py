import os
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from parity_transformer import main_transformer
from parity_patcher import main_patcher
from parity_math import main_math


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        if cmd == "math":
            main_math()
        elif cmd == "patcher":
            main_patcher()
        elif cmd == "transformer":
            main_transformer()
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python parity.py [math|patcher|transformer] [options]")
    else:
        main_math()
        main_patcher()
        main_transformer()