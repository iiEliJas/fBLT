# fBLT

[![CI](https://github.com/iiEliJas/fBLT/actions/workflows/ci.yml/badge.svg)](https://github.com/iiEliJas/fBLT/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Fast Byte Latent Transformer in pure C and CUDA. A byte-level language model with no tokenizer, built for code completion and small LLMs.

Based on two papers from Meta:
- [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) - Byte modeling with entropy-based dynamic patching. Matches token-based LLM scaling, no vocabulary needed.
- [Fast Byte Latent Transformer](https://arxiv.org/abs/2605.08044) - Faster inference with diffusion decoding and self-speculation.

## How it works

Five-stage pipeline:

1. **Entropy Model**: small byte-level LM that predicts per-byte entropy used for patching

2. **Patcher**: chops the byte stream into patches using entropy thresholds (rule-based so no learning)

3. **Local Encoder**: tiny transformer that compresses byte patches into embeddings, using hash n-gram embeddings to recognize byte patterns without a vocabulary

4. **Patch Transformer**: the main model. Block-causal attention over patches.

5. **Local Decoder**: tiny transformer that expands patches back to bytes, with optional self-speculation

## Build

```bash
make test         # build + run tests
make train-blt-d  # build train_blt_d executable to train
make main         # build main executable
make info         # show build config
make clean        # remove obj/, bin/
make help         # show all targets
```

Default is `gcc -O2 -std=c99 -Wall -Wextra`. Change it:

```bash
make CC=clang CFLAGS="-O3 -std=c99 -Wall"
```

CUDA: `make CUDA=1 <target>` compiles `.cu` files with nvcc, outputs to `obj-cuda/` and `bin-cuda/`.

## Layout

```
csrc/
  core/               Core: tensor, memory, backend, allocator, JSON, tools
  ops/                Op signatures + CPU/CUDA implementations
  models/             Model components
  infer/              Inference: KV cache, speculation, block generation

fblt/
  scripts/            Python utilities

tests/                Test suite
bench/                Benchmarks
run/                  Entry points
configs/              Model configs (JSON)
data/                 Sample data (tests, training, etc.)
```

## Training configuration

The production config for 2.97M-parameter BLT-D training (embed=192, hidden=384, 2/2/2 layers):

```bash
./bin-cuda/train_blt_d --backend cuda \
  --embed 192 --hidden 384 --layers 2 \
  --steps 40000 --lr 0.05 --lr-decay 1 \
  --mask-warmup 33200 --mask-scale 0.3 \
  --t-min 0.1 --t-warmup-hi 0.25 --t-hi-start 0.8 \
  --cross-attn all --optimizer sgd \
  --diffusion 1 --block-size 4 --window 48 \
  --save-weights runs/my_checkpoint.fblt
```

**What was tested and found** (see [ABLATIONS.md](docs/ABLATIONS.md) for details):

| Setting | Value | Finding |
|---|---|---|
| Step budget | 40k | Safe convergence floor; masked acc > 0.99 across seeds |
| Mask-warmup | 83% of steps | 95% is a minor improvement, not required |
| Decoder depth | 2 layers | Deeper decoder (3-4 layers) doesn't help |
| Encoder depth | 2 layers | Deeper encoder shows instability |
| Cross-attn | all or last | xlast is ~15% faster, quality indistinguishable |
| High-t warmup | on | -0.33 BPB improvement, no instability |
| Mask-late | off | No benefit found |

CUDA delivers **76x training speedup** and **34x generation speedup** over CPU at production config (E=192, H=384, 2.97M params). CPU baseline is single-threaded naive loops. Matmul fp32 hits **6.4 TFLOP/s** on the desktop RTX 4060 (**42.4% MFU** against the 15.11 TFLOP/s spec peak). BF16 mixed-precision matmuls reach **~20 TFLOP/s** (~3x the fp32 rate).

```bash
# Training
./bin-cuda/train_blt_d --backend cuda --embed 192 --hidden 384 \
  --steps 40000 --lr 0.05 --mask-scale 0.3 \
  --t-warmup-hi 0.25 --t-hi-start 0.8

# Inference
./bin-cuda/infer_bench --backend cuda --plain runs/my_checkpoint.fblt \
  --bltd runs/my_checkpoint.fblt --prompts 8 --new-bytes 64
```

Full results: [BENCHMARKS.md](docs/BENCHMARKS.md).

**OPEN items:** Seed-dependent non-determinism from CUDA atomics. Acceptance-rate root cause at this scale is unknown.

## Inference

Three inference modes, benchmarked on the 2.97M-param checkpoint:

- **BLT-S**: draft k bytes with decoder-only passes, verify with one full forward. Byte-identical output, fewer encoder/global passes.
- **BLT-D**: decoder generates a whole block of future bytes in parallel from masked states. Cheapest per byte, but drafts drift.
- **BLT-DV**: BLT-D drafts, then the model verifies them. Output matches greedy so the draft just makes it cheaper.

**Headline finding:** At 2.97M params, all verified inference methods (BLT-S, BLT-DV) cost *more* memory bandwidth than greedy. BLT-DV acceptance rates (1.6-5.2%) are 18-42x lower than the paper's 3B results. Root cause likely because of the 340x scale gap. See [BENCHMARKS.md](docs/BENCHMARKS.md) for full results.

![Speculative acceptance rates by method](graphs/acceptance.png)
*Drafted-byte acceptance rate by method, 2.97M-param checkpoint. BLT-S k=4 leads at 31%, declining sharply with k. BLT-DV variants cluster at 2–25%.*

![Mean wall-clock latency per prompt](graphs/latency.png)
*Mean wall-clock ms per prompt (64 new bytes). KV-cache usage differs between inference paths, so these are rough wall-clock numbers, not a clean comparison.*

```bash
make bench-infer                       # needs runs/*_40k.fblt checkpoints
python3 fblt/scripts/bench_plots.py           # writes graphs/*.png
```

## Ablation sweeps

Architecture and schedule ablations at 2.97M parameters (embed=192, hidden=384, 2/2/2 layers) on ~67 MB of C source code. Each axis tested with 3-6 seeds at 40k steps.

| Axis | Finding |
|---|---|
| Decoder depth | Deeper decoder (3-4 layers) doesn't help so 2/2/2 is optimal |
| Encoder depth | Deeper encoder (3/2/2) is harmful so 2 layers is best |
| Cross-attn | base (all) and xlast (last) tied at 83% warmup - xlast ~15% faster |
| High-t warmup | Improves BPB by 0.33 (2.7x noise floor), no instability |
| Mask-late | No benefit found |
| Step budget | 40k is safe convergence floor (cliff sits somewhere at 25k-35k) |

Full tables, raw numbers, and production config: [ABLATIONS.md](docs/ABLATIONS.md).

## TODO

**Done:**
- Tensor and memory system (arena allocator, shapes, strides)
- Backend dispatch (CPU/CUDA abstraction)
- Core ops on CPU + CUDA (elementwise, matmul, softmax, attention, etc.)
- Entropy model, patcher
- Attention (causal, cross-attention, custom masks)
- Full test harness (unit, integration, golden files)
- Hash n-gram embeddings
- Local encoder + decoder
- Patch transformer (global model)
- End-to-end forward pass
- FLOP counting and profiling
- Ablation sweeps (BPB tables)
- CUDA backend
- Training configuration validation (2.97M params, 40k steps)
- Inference re-verification (acceptance rates, bandwidth analysis)

**Todo:**
- Multi-GPU / larger-scale training runs
- Windows and MacOS support
- Inference improvements
- Test acceptance rates on larger checkpoints (root-cause investigation)

## Docs

- [API_REFERENCE.md](API_REFERENCE.md) - Tensor and op API
- [BENCHMARKS.md](docs/BENCHMARKS.md) - Full benchmark report
- [ABLATIONS.md](docs/ABLATIONS.md) - Ablation sweep results
- [TEST_API_REFERENCE.md](TEST_API_REFERENCE.md) - Writing tests

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch policy, code style, and PR guidelines.

## License

[Apache 2.0](LICENSE)
