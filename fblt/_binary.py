import os
import sys
import typing
from pathlib import Path


def _candidate_dirs(backend: str) -> typing.List[Path]:
    build_dir = "build-cuda" if backend == "cuda" else "build"
    dirs = []
    env = os.environ.get("FBLT_BIN_DIR")
    if env:
        dirs.append(Path(env))
    dirs.append(Path(__file__).resolve().parent / "bin")
    repo_root = Path(__file__).resolve().parent.parent
    dirs.append(repo_root / build_dir)
    dirs.append(Path.cwd() / build_dir)
    return dirs


def resolve_binary(name: str, backend: str) -> str:
    suffix = ".exe" if sys.platform == "win32" else ""
    searched = []
    for d in _candidate_dirs(backend):
        path = d / (name + suffix)
        searched.append(str(path))
        if path.is_file():
            if sys.platform != "win32" and not os.access(path, os.X_OK):
                continue
            return str(path)

    build_dir = "build-cuda" if backend == "cuda" else "build"
    raise FileNotFoundError(
        "Binary not found: {}. Searched: {}. Build it with: cmake --build {} --target {}, "
        "or set FBLT_BIN_DIR.".format(name, ", ".join(searched), build_dir, name)
    )
