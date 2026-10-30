import glob
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


def ensure_wsl_cuda_env() -> None:
    """On WSL2, an apt-installed libnvidia-ptxjitcompiler shadows the
    matching WSL driver JIT and crashes CUDA binaries. Put the WSL driver
    directory first on LD_LIBRARY_PATH when it is available."""
    if sys.platform != "linux":
        return
    driver_dirs = glob.glob("/usr/lib/wsl/drivers/nvmdsi.inf_amd64_*")
    if not driver_dirs:
        return
    d = driver_dirs[0]
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in cur.split(":") if p]
    if d not in parts:
        parts.insert(0, d)
    # wsl/lib holds libcuda; keep it present
    for extra in ("/usr/lib/wsl/lib",):
        if extra not in parts:
            parts.append(extra)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


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
