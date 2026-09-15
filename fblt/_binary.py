import os
import sys
from pathlib import Path

_NAME_TO_MAKE_TARGET = {
    "train_blt_d": "train-blt-d",
    "infer": "infer",
}


def resolve_binary(name: str, backend: str) -> str:
    if backend == "cuda":
        path = Path("bin-cuda") / name
    else:
        path = Path("bin") / name

    # windows needs the .exe suffix
    if sys.platform == "win32":
        path = path.with_suffix(".exe")

    path_str = str(path)
    target = _NAME_TO_MAKE_TARGET[name]

    if not os.path.isfile(path_str):
        cuda_prefix = "CUDA=1 " if backend == "cuda" else ""
        raise FileNotFoundError(
            f"Binary not found: {path_str}. Build with: make {cuda_prefix}{target}"
        )

    # os.X_OK is unreliable on Windows
    if sys.platform != "win32" and not os.access(path_str, os.X_OK):
        cuda_prefix = "CUDA=1 " if backend == "cuda" else ""
        raise FileNotFoundError(
            f"Binary not executable: {path_str}. Rebuild with: make {cuda_prefix}{target}"
        )

    return path_str
