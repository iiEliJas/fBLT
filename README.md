# FBLT

Fast Byte Latent Transformer in pure C. A byte-level language model with no tokenizer, built for code completion and small LLMs.

**Status: ~50% done.** Core infrastructure, operators, and components are done. Missing training loop, CUDA kernels, and inference speedups.

Based on two papers from Meta:
- [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) - Direct byte modeling with entropy-based dynamic patching. Matches token-based LLM scaling, no vocabulary needed.
- [Fast Byte Latent Transformer](https://arxiv.org/abs/2605.08044) - Faster inference via diffusion decoding and self-speculation.

## Why no tokenizer?

Tokenizers have limitations. They force a fixed vocabulary on your data before you even start training. This causes problems:

- Sensitivity to noise and out-of-domain text
- Weird multilingual and low-resource behavior
- Lost character information your model could have learned
- Compute/quality tradeoff baked in at the tokenizer level

FBLT works directly on bytes. The trick: entropy-based patching. High-entropy (hard to predict) bytes get short patches and more compute. Predictable bytes get long patches and run cheap. This gives you adaptive compute allocation without any tokenizer baggage.

Result: better scaling, robustness, and no vocabulary constraints.

## How it works

Five-stage pipeline:

1. **Entropy Model** - Small byte-level LM that predicts per-byte entropy. Just guides patching, trained first.

2. **Patcher** - Uses entropy thresholds to slice the byte stream into variable-length patches. Simple rule-based, no learning.

3. **Local Encoder** - Tiny transformer that compresses byte patches into embeddings. Uses hash n-gram embeddings (rolling polynomial hash) to recognize byte patterns without a vocabulary table.

4. **Patch Transformer** - The actual big model. Does block-causal attention over patches (sequence is 4-8x shorter than bytes). This is where compute happens.

5. **Local Decoder** - Tiny transformer that expands patches back to bytes, autoregressively. Can draft multiple bytes and verify in one go (self-speculation).

## Design

Pure C + CUDA.

**One interface, two backends.** Every op (matmul, attention, ...) lives in a header. Implemented once for CPU, once for CUDA. Model code never knows the difference.

**CPU is the ground truth.** Every new op gets a simple C implementation first. CUDA versions validate against it.

**Config, not code.** All the BLT paper ablations (hash sizes, cross-attention placement, layer splits) are JSON config values. No recompiling to try things.

## Build

Use make.

```bash
make test       # build + run tests
make main       # build main executable
make info       # show build config
make clean      # remove obj/, bin/
make help       # show all targets
```

Default is `gcc -O2 -std=c99 -Wall -Wextra`. Change it:

```bash
make CC=clang CFLAGS="-O3 -std=c99 -Wall"
```

CUDA isn't wired up to the Makefile yet.

## Layout

```
include/blt/            Public headers
  ├── core/             Tensor, memory, backend dispatch
  ├── ops/              op signatures
  └── models/           Model component signatures

src/
  ├── core/             Implementation
  ├── backend_cpu/      CPU ops
  ├── models/           Model components

tests/
  ├── unit/             op tests
  ├── integration/      Full pipeline tests
  ├── test_main.c       Entry point
  └── test_helpers.c    Asserts and utilities

run/                    Entry points
configs/                Model configs (JSON)
```

## What's done, what's next

**DONE:**
- Tensor and memory system (arena allocator, shapes, strides)
- Backend dispatch (CPU/CUDA abstraction)
- Basic ops (elementwise, reductions, matmul, etc.) on CPU
- Entropy model (byte-level LM for patching)
- Patcher (entropy thresholds, monotonicity constraints)
- Attention (causal, cross-attention, custom masks)
- Full test harness (unit, integration, golden files)

**TODO:**
- Hash n-gram embeddings (rolling poly hash, tables)
- Local encoder + decoder (interleaved attention + transformers)
- Patch transformer (global model, block-causal)
- End-to-end forward pass validation
- Config file plumbing
- CUDA kernels
- Training loop
- Inference/generation
- Self-speculation (BLT-S) and diffusion decoding (BLT-D)
- Benchmarking tools
- Multi-GPU

## Docs

- `API_REFERENCE.md` - Tensor and op API
- `TEST_API_REFERENCE.md` - Writing tests