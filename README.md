# fBLT

[![CI](https://github.com/iiEliJas/fBLT/actions/workflows/ci.yml/badge.svg)](https://github.com/iiEliJas/fBLT/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Fast Byte Latent Transformer in pure C and CUDA. A byte-level language model with no tokenizer, built for code completion and small LLMs.

Based on two papers from Meta FAIR (with Stanford and University of Washington collaborators):
- [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) - Byte modeling with entropy-based dynamic patching. Matches token-based LLM scaling, no vocabulary needed.
- [Fast Byte Latent Transformer](https://arxiv.org/abs/2605.08044) - Faster inference with diffusion decoding and self-speculation.

**Status**: research prototype

## Table of contents

- [How it works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [Build](#build)
- [Layout](#layout)
- [Training configuration](#training-configuration)
- [Inference](#inference)
- [Benchmarks](#benchmarks)
- [Community experiments](#community-experiments)
- [Issues](#issues)
- [TODO](#todo)
- [Docs](#docs)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

## How it works

Five-stage pipeline:

1. **Entropy Model**: small byte-level LM that predicts per-byte entropy used for patching

2. **Patcher**: chops the byte stream into patches using entropy thresholds (rule-based so no learning)

3. **Local Encoder**: tiny transformer that compresses byte patches into embeddings, using hash n-gram embeddings to recognize byte patterns without a vocabulary

4. **Patch Transformer**: the main model. Block-causal attention over patches.

5. **Local Decoder**: tiny transformer that expands patches back to bytes, with optional self-speculation

## Prerequisites

- C compiler (`gcc` or `clang`), C99-compatible
- CMake >= 3.18
- Python >= 3.9
- (Optional, for GPU training/inference) CUDA toolkit with `nvcc`

## Quickstart

```bash
pip install -e .                    # install fblt package + entry points

# Build C binaries
cmake -S . -B build                  # CPU build
cmake --build build -j$(nproc)       # build all targets

# Or for CUDA:
cmake -S . -B build-cuda -DUSE_CUDA=ON
cmake --build build-cuda -j$(nproc)

python fblt/scripts/build_sample_corpus.py  # generate sample training data from repo source

# Train a tiny model (debug config, 10 steps)
fblt-train --config configs/train/debug.yaml --backend cpu

# Generate text from the trained checkpoint (auto-shapes from resolved_config.yaml)
fblt-infer --checkpoint "runs/debug-*/checkpoint.fblt" --backend cpu --prompt "int main"
```

The sample corpus is generated from the repo's own `.c`/`.h` source files (~1.1 MB) under the Apache 2.0 license. The debug config trains a small model in seconds. For production training, use `configs/train/production.yaml` with your own larger corpus (e.g., the stack-smol-C-derived dataset used for the benchmark results).

### Using your own dataset

The training binary expects a flat `.bin` file of raw bytes with no header, no framing, no length prefix. If you're using multiple source files, separate them with a newline (`\n`).

To use your own data, place the `.bin` file in the repo and point the config at it:

```bash
# Option 1: edit the config
# configs/train/production.yaml:
#   corpus: /path/to/your/train.bin

# Option 2: override on the command line
fblt-train --config configs/train/production.yaml --backend cpu \
  --override corpus=/path/to/your/train.bin
```

For heldout evaluation, set `--eval-corpus` to a separate `.bin` file not overlapping with your training data. The inference benchmark (`infer_bench`) defaults to `data/heldout.bin` for this purpose.

## Build

```bash
# CPU build
cmake -S . -B build -DUSE_CUDA=OFF
cmake --build build -j$(nproc)                 # build all targets
cmake --build build --target test              # build + run C test suite

# CUDA build
cmake -S . -B build-cuda -DUSE_CUDA=ON
cmake --build build-cuda -j$(nproc)            # build all targets
cmake --build build-cuda --target test         # build + run C test suite

# Other useful targets
cmake --build build --target main              # build main executable
cmake --build build --target train_blt_d       # build trainer
cmake --build build --target infer             # build inference binary
cmake --build build --target sandbox_run       # build + run sandbox
cmake --build build --target bench-infer       # build + run inference benchmark
```

Default is `gcc -O2 -std=c99 -Wall -Wextra`. Change it:

```bash
cmake -S . -B build -DCMAKE_C_COMPILER=clang -DCMAKE_C_FLAGS="-O3 -std=c99 -Wall"
```

CUDA: `cmake -S . -B build-cuda -DUSE_CUDA=ON` enables CUDA support and compiles `.cu` files with `nvcc`. The default architecture is `89`; configure another GPU with `-DCMAKE_CUDA_ARCHITECTURES=86` or the value required by your device. If `nvcc` is installed outside `/usr/local/cuda`, pass `-DCUDAToolkit_ROOT=/path/to/cuda` or `-DCMAKE_CUDA_COMPILER=/path/to/nvcc`.

On WSL2, if an apt NVIDIA library shadows the WSL driver, put the WSL driver directory first on `LD_LIBRARY_PATH` and set `CUDA_VISIBLE_DEVICES=0` if the variable is empty.

### Python setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .                      # install fblt package + entry points
pip install ruff pytest               # dev dependencies
```

Requires Python >= 3.9.

Run Python tests and lint checks:

```bash
pytest tests/                         # run Python test suite
ruff check .                          # lint
ruff format --check .                 # format check
```

For the C test suite parity tests, generate golden data first:

```bash
pip install torch numpy
python tests/py/parity_generate.py  # or: cmake --build build --target parity-data
cmake --build build --target test   # build + run C tests
```

## Layout

```
csrc/
  core/               Core: tensor, memory, backend, allocator, JSON, tools
  ops/                Op signatures + CPU/CUDA implementations
  models/             Model components
  infer/              Inference: KV cache, speculation, block generation

fblt/                 Python package
  __init__.py
  train.py            Training entry point (fblt-train)
  infer.py            Inference entry point (fblt-infer)
  _binary.py          Binary path resolution
  _config.py          TrainConfig / InferConfig dataclasses, YAML loader
  _runner.py          Subprocess runner with live streaming
  scripts/            Utilities (bench_plots, bench_report, gen_configs, prep_corpus, build_sample_corpus)

configs/
  train/              Training YAML configs (production, debug)
  infer/              Inference YAML configs (greedy, selfspec, blockdiff, blockdv)
  ablations/          Ablation sweep configs (JSON)

tests/                Test suite (C + Python)
bench/                Benchmarks
run/                  C entry points (train_blt_d.c, infer.c, etc.)
data/                 Training data (sample corpus, golden test tensors)

build/                CMake build output (CPU)
build-cuda/           CMake build output (CUDA, when -DUSE_CUDA=ON)
```

## Training configuration

An example for the production config for 2.97M-parameter BLT-D training (embed=192, hidden=384, 2/2/2 layers):

```bash
fblt-train --config configs/train/production.yaml --backend cuda
```

The wrapper creates a run directory under `runs/`, writes a `resolved_config.yaml` with the full config, and calls `train_blt_d`. All `--override key=value` flags are passed through, and `--backend` is always required on the command line (never in YAML).

The underlying C binary can still be called directly:

```bash
./build-cuda/train_blt_d --backend cuda \
  --embed 192 --hidden 384 --layers 2 \
  --steps 40000 --lr 0.05 --lr-decay 1 \
  --mask-warmup 33200 --mask-scale 0.3 \
  --t-min 0.1 --t-warmup-hi 0.25 --t-hi-start 0.8 \
  --cross-attn all --optimizer sgd \
  --diffusion 1 --block-size 4 --window 48 \
  --save-weights runs/my_checkpoint.fblt
```

## Inference

### Generating text from a checkpoint

Use `fblt-infer` to generate text from a trained model:

```bash
fblt-infer --checkpoint runs/debug_checkpoint.fblt --backend cpu \
  --config configs/infer/greedy.yaml --prompt "int main"
```

If the checkpoint was produced by `fblt-train`, the wrapper auto-detects model dimensions from `resolved_config.yaml`. So no need to pass `--embed`, `--hidden`, etc. manually. For checkpoints not produced by the wrapper, pass shape flags via `--override`:

```bash
fblt-infer --checkpoint my_model.fblt --backend cpu \
  --override embed=192 --override hidden=384 \
  --override enc-layers=2 --override glob-layers=2 --override dec-layers=2 \
  --prompt "def "
```

Inference methods: `greedy` (default), `selfspec`, `blockdiff`, `blockdv`. See `configs/infer/` for example YAML configs for each method.

## Benchmarks
Full results: [BENCHMARKS.md](docs/BENCHMARKS.md).

CUDA delivers **76x training speedup** and **34x generation speedup** over CPU at production config (E=192, H=384, 2.97M params). CPU baseline is single-threaded naive loops. Matmul fp32 hits **6.4 TFLOP/s** on the desktop RTX 4060 (**42.4% MFU** against the 15.11 TFLOP/s spec peak). BF16 mixed-precision matmuls reach **~20 TFLOP/s** (~3x the fp32 rate).

`infer_bench` is a separate research/benchmarking tool that compares inference methods and writes metrics to `bench/results.jsonl`.

```bash
cmake --build build --target bench-infer  # needs runs/*_40k.fblt checkpoints
python3 fblt/scripts/bench_plots.py    # writes graphs/*.png
```

Three inference modes, benchmarked on the 2.97M-param checkpoint:

- **BLT-S**: draft k bytes with decoder-only passes, verify with one full forward. Byte-identical output, fewer encoder/global passes.
- **BLT-D**: decoder generates a whole block of future bytes in parallel from masked states. Cheapest per byte, but drafts drift.
- **BLT-DV**: BLT-D drafts, then the model verifies them. Output matches greedy so the draft just makes it cheaper.

**Headline finding:** At 2.97M params, all verified inference methods (BLT-S, BLT-DV) cost *more* memory bandwidth than greedy. BLT-DV acceptance rates (1.6-5.2%) are 18-42x lower than the paper's 3B results. Root cause likely because of the 340x scale gap.

![Speculative acceptance rates by method](graphs/acceptance.png)
*Drafted-byte acceptance rate by method, 2.97M-param checkpoint. BLT-S k=4 leads at 31%, declining sharply with k. BLT-DV variants cluster at 2-25%.*

![Mean wall-clock latency per prompt](graphs/latency.png)
*Mean wall-clock ms per prompt (64 new bytes). KV-cache usage differs between inference paths, so these are rough wall-clock numbers, not a clean comparison.*

### Ablation sweeps

Architecture and schedule ablations at 2.97M parameters (embed=192, hidden=384, 2/2/2 layers) on ~67 MB of C source code. Each axis tested with 3-6 seeds at 40k steps.

| Setting | Value | Finding |
|---|---|---|
| Step budget | 40k | Safe convergence floor; masked acc > 0.99 across seeds (cliff sits somewhere at 25k-35k) |
| Mask-warmup | 83% of steps | 95% is a minor improvement, not required |
| Decoder depth | 2 layers | Deeper decoder (3-4 layers) doesn't help, 2/2/2 is optimal |
| Encoder depth | 2 layers | Deeper encoder (3/2/2) is harmful, 2 layers is best |
| Cross-attn | all or last | base (all) and xlast (last) tied at 83% warmup, xlast ~15% faster |
| High-t warmup | on | Improves BPB by 0.33 (2.7x noise floor), no instability |
| Mask-late | off | No benefit found |

Full tables, raw numbers, and production config: [ABLATIONS.md](docs/ABLATIONS.md).

## Community experiments

fBLT is small and I have more ideas than time to test them. If you change
something and see what happens, tell me about it. It does not matter if it
worked or not.

Try a different hyperparameter, run it on another corpus, push it to a
bigger scale, or look into the CUDA non-determinism from Known issues. You
do not need a plan. Just curiosity and a GPU, or a patient CPU.

Open an issue with what you changed, your hardware, and the command you
used. Attach the `resolved_config.yaml` from your run directory so others
can repeat it. If you have numbers, add them even if they are rough: BPB,
acceptance rate, wall-clock, anything.

Negative results count too. They save the next
person from trying it too ;D

## Issues

- Seed-dependent non-determinism from CUDA atomics.

- Acceptance-rate root cause at this scale is unknown.

- [CLOSED] [last_row_gap.md](docs/experiments/last_row_gap.md) Last-row accuracy inside each training window sits far below interior-row accuracy (~40% vs ~98%). 
Two Hypotheses are being considered. Either the last row gets far less supervision per window than interior rows.
Or patches that get cut off at the window edge are harder for the model to represent. 

- [forced_boundary_patches.md](docs/experiments/forced_boundary_patches.md) Open issue on comparing forced-boundary patches to natural closed patches.

- [patch_cap_limit.md](docs/experiments/patch_cap_limit.md) The 128-patch cap is another open issue. I added it because BLT-D did not support more patches at the time. Raising it and measuring the effect is an open experiment i have not yet done.

## TODO
- Inference improvements
- Multi-GPU / larger-scale training runs
- MacOS support
- Test acceptance rates on larger checkpoints (root-cause investigation)

## Docs

- [BENCHMARKS.md](docs/BENCHMARKS.md) - Full benchmark report
- [ABLATIONS.md](docs/ABLATIONS.md) - Ablation sweep results
- [CLI_REFERENCE.md](docs/CLI_REFERENCE.md) - CLI commands for training and inference
- [last_row_gap.md](docs/experiments/last_row_gap.md) - Closed investigation into the last-row accuracy gap

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch policy, code style, and PR guidelines.

## References

If you build on this work, please consider citing the foundational Meta papers:

```bibtex
@misc{pagnoni2024bytelatenttransformerpatches,
      title={Byte Latent Transformer: Patches Scale Better Than Tokens},
      author={Artidoro Pagnoni and Ramakanth Pasunuru and Pedro Rodriguez and John Nguyen and
              Benjamin Muller and Margaret Li and Chunting Zhou and Lili Yu and Jason Weston and
              Luke Zettlemoyer and Gargi Ghosh and Mike Lewis and Ari Holtzman and Srinivasan Iyer},
      year={2024},
      eprint={2412.09871},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      doi={10.48550/arXiv.2412.09871}
}

@misc{kallini2026fastbytelatenttransformer,
      title={Fast Byte Latent Transformer},
      author={Julie Kallini and Artidoro Pagnoni and Tomasz Limisiewicz and Gargi Ghosh and
              Luke Zettlemoyer and Christopher Potts and Xiaochuang Han and Srinivasan Iyer},
      year={2026},
      eprint={2605.08044},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

[Apache 2.0](LICENSE)
