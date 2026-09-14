# Contributing

Thanks for your interest in contributing to fBLT.

## Branches

- **`main`** is the stable branch. It always builds and passes `make test`.
- **`dev`** is the integration branch. Open your PR against `dev`.

## Local checks

Run the lightweight checks locally before submitting:

```sh
make test
python3 tests/py/test_config.py
python3 tests/py/test_entrypoints.py
```

This builds the project and runs the full test suite (unit + integration tests). It requires `gcc` and `make`. No CUDA needed for the default CPU path.

For CUDA changes, additionally check on a CUDA-capable host:

```sh
make CUDA=1 test
```

## Code style

- C99, `gcc -O2 -std=c99 -Wall -Wextra` (default Makefile)
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
