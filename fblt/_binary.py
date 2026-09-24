import os
import sys
from pathlib import Path


def resolve_binary(name: str, backend: str) -> str:
    if backend == "cuda":
        path = Path("build-cuda") / name
    else:
        path = Path("build") / name

    # windows needs the .exe suffix
    if sys.platform == "win32":
        path = path.with_suffix(".exe")

    path_str = str(path)
    build_dir = "build-cuda" if backend == "cuda" else "build"

    if not os.path.isfile(path_str):
        raise FileNotFoundError(
            f"Binary not found: {path_str}. "
            f"Build it with: cmake --build {build_dir} --target {name}"
        )

    # os.X_OK is unreliable on Windows
    if sys.platform != "win32" and not os.access(path_str, os.X_OK):
        raise FileNotFoundError(
            f"Binary not executable: {path_str}. "
            f"Build it with: cmake --build {build_dir} --target {name}"
        )

    return path_str
