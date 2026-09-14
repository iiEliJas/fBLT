import os
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from parity_transformer import main_transformer
from parity_patcher import main_patcher
from parity_math import main_math
from parity_global_tranformer import main_global_transformer
from parity_local_decoder import main_local_decoder
from parity_local_encoder import main_local_encoder


if __name__ == "__main__":
    main_math()
    main_patcher()
    main_transformer()
    main_global_transformer()
    main_local_decoder()
    main_local_encoder()