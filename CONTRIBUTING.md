# Contributing

Thanks for your interest in contributing to fBLT.

## Branches

- **`main`** is the stable branch. It always builds and passes the CMake test target. Open your PR against `main`.

## Local checks

Run the lightweight checks locally before submitting:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install ruff pytest

ruff check .
ruff format --check .
pytest tests/

# C tests
cmake -S . -B build -DUSE_CUDA=OFF
cmake --build build --target test
```

This builds the project and runs the full C test suite (unit + integration tests). It requires CMake and a C99 compiler. No CUDA is needed for the default CPU path.

For CUDA changes, check a CUDA-capable host as well:

```sh
cmake -S . -B build-cuda -DUSE_CUDA=ON
cmake --build build-cuda --target test
cmake --build build-cuda --target cuda-smoke
```

CUDA builds default to architecture `89`. Use `-DCMAKE_CUDA_ARCHITECTURES=...` for another GPU, and use `-DCUDAToolkit_ROOT=...` or `-DCMAKE_CUDA_COMPILER=...` when the toolkit is installed outside `/usr/local/cuda`.

On WSL2, if an apt NVIDIA library shadows the WSL driver, put the WSL driver directory first on `LD_LIBRARY_PATH` and set `CUDA_VISIBLE_DEVICES=0` if the variable is empty.

## Code style

- C99, `gcc -O2 -std=c99 -Wall -Wextra` (CMake defaults)
- Python: Ruff formatter (line-length 100, target Python 3.11)
- 4-space indentation, 120-column limit for C/CUDA files
- Match the patterns in the file you are editing

## Performance PRs

Performance PRs should include:
- The exact build command and commit hash
- Hardware details (CPU, RAM, GPU if applicable)
- The benchmark command and raw output
- At least 3 runs with median throughput

## What to include in a PR

- Focused change (one concern per PR)
- Tests if adding new functionality
- Updated docs if changing user-facing behavior

## Reporting issues

Open an issue with:
- Your OS, compiler, and CPU
- Steps to reproduce
- Expected vs actual behavior
- Relevant log output
