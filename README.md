# FBLT

Fast Byte Latent Transformer in pure C and CUDA. A byte-level language model with no tokenizer, built for code completion and small LLMs.

**Status: ~80% done.** Full CPU pipeline: ops, models, training loop, and all three Fast-BLT inference modes (BLT-S / BLT-D / BLT-DV) with benchmarked results below. Missing: CUDA kernels.

Based on two papers from Meta:
- [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) - Direct byte modeling with entropy-based dynamic patching. Matches token-based LLM scaling, no vocabulary needed.
- [Fast Byte Latent Transformer](https://arxiv.org/abs/2605.08044) - Faster inference via diffusion decoding and self-speculation.

## Why no tokenizer?

Tokenizers fix a vocabulary before training, which causes noise sensitivity, poor multilingual/low-resource behavior, and lost character information. The compute/quality tradeoff is baked in at the tokenizer level.

FBLT works directly on bytes. **Entropy-based patching** allocates compute dynamically: predictable bytes get long patches (cheap), hard-to-predict bytes get short patches (more compute).

Result: better scaling, robustness, and no vocabulary constraints.

## How it works

Five-stage pipeline:

1. **Entropy Model** - small byte-level LM that predicts per-byte entropy to guide patching

2. **Patcher** - slices the byte stream into variable-length patches using entropy thresholds (rule-based, no learning)

3. **Local Encoder** -  tiny transformer that compresses byte patches into embeddings, using hash n-gram embeddings to recognize byte patterns without a vocabulary

4. **Patch Transformer** - the main model. Block-causal attention over patches (4-8x shorter than the raw byte sequence).

5. **Local Decoder** - tiny transformer that expands patches back to bytes autoregressively, with optional self-speculation

## Design

Pure C + CUDA.

- **One interface, two backends.** Every op lives in a header, implemented once for CPU and once for CUDA. Model code is backend-agnostic.
- **CPU is the ground truth.** Each op gets a C implementation first. CUDA versions validate against it.
- **Config-driven.** All architecture options are JSON config values. (hash sizes, cross-attention placement, layer splits)

## Build

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
  |── bench/            Benchmarks
  ├── unit/             op tests
  ├── integration/      Full pipeline tests
  |── py/               Python parity generations
  ├── test_main.c       Entry point
  └── test_helpers.c    Asserts and utilities

run/                    Entry points
configs/                Model configs (JSON)
data/                   Sample data (tests, training, etc.)
tools/                  Scripts and utilities
```


## Ablation sweeps

I ran a first round of architecture ablations on the C code subset of
the-stack-smol (~67 MB train / ~3.7 MB held-out, byte-level) with only the CPU-backend. Each config
trains for a fixed budget (2000 steps, ~17-20 min on my CPU) and reports
held-out BPB. Protocol, tables and notes live in `docs/ablations.md`,
raw numbers in `bench/results.jsonl`.

Whats interesting: at this scale, doing the opposite of what the paper suggests
worked better -

| config | held-out BPB | throughput |
|---|---|---|
| baseline (paper defaults) | 5.3925 | 290 B/s |
| **winner**: tiny ngram tables, decoder-only cross-attn, fixed-stride-4 patches | **5.3782 (-0.27%)** | **426 B/s (+47%)** |

Bigger ngram hash tables never helped, encoder cross-attention placement
can't matter when the decoder doesn't read patches, and dumb fixed
patching beats entropy-based dynamic patching on throughput for almost no
quality cost. Reproduce with `make sweep` + the configs in `configs/ablations/`.

![winner vs best of each sweep](docs/winner_vs_best.png)

The n-gram sweep is a good example of "more is not better" - every bigger
table loses to the tiny `{3,4} @ 50k` baseline, and dropping n-gram entirely
costs only +0.07% BPB:

| config | BPB | dBPB | throughput |
|---|---|---|---|
| **baseline `{3,4} @ 50k`** | **5.3925** | / | 290 B/s |
| none | 5.3961 | +0.07% | 290 B/s |
| `{3,4,5}` @ 200k | 5.3933 | +0.01% | 266 B/s |
| `{6,7,8}` @ 100k | 5.3932 | +0.01% | 290 B/s |
| all `{3..8}` @ 100k | 5.3980 | +0.10% | 272 B/s |

![ngram comparison](docs/ngram_comparison.png)

![BPB vs throughput](docs/patch_bpb_vs_throughput.png)



## Training and inference results

Three ways to generate faster than one-full-forward-per-byte, all
benchmarked on two 40k-step models trained on ~67 MB of C code (~300k
params each):

- **BLT-S (self-speculation)** - draft k bytes with decoder-only
  passes, verify with one full forward. Byte-identical output, big
  savings on encoder/global passes.
- **BLT-D (block diffusion)** - the decoder generates a whole block of
  B future bytes in parallel from masked states, unmasking confident
  positions first. Cheapest per byte, but drafts drift from what greedy
  decoding would produce.
- **BLT-DV (diffusion + verification)** - BLT-D drafts, then the same
  model verifies them autoregressively. Output matches greedy; the
  draft just makes it cheaper.

`nfe` = one forward pass of a submodule. `agreement` = fraction of
generated bytes matching the same model's greedy output.

### What the training experiments showed

- The two objectives genuinely interfere: mask-loss weight 1.0 (paper
  default) costs **+2.1 BPB** over plain BLT; a warmup + 0.3 cap
  recovers most of it (**1.65 vs 1.26 BPB**) while the model still
  predicts masked future blocks at ~100% top-1.
- With the mask loss silenced, the full BLT-D pipeline lands within
  noise of the plain arm (2.13 vs 2.07 BPB) - the machinery is exactly
  as good as plain training when the objective matches.
- The `1/t` loss weight needs a timestep floor (`t >= 0.05`), or rare
  draws produce weights up to ~10^6 and inf spikes.

### Quality vs cost

![quality frontier](docs/nfe_quality_frontier.png)

Everything on the agreement=1.0 line is exactly greedy quality. BLT-DV
sits at ~0.66-0.81 decoder passes per byte; raw BLT-D drafts are 10x
cheaper but diverge (agreement 0.08-0.14) - useful for diverse
sampling, not for exact continuation.

### Where the passes go

![nfe map](docs/enc_dec_map.png)

BLT-S trades *more* decoder passes for *far fewer* encoder+global
passes (1.0 -> 0.21 at k=16) - it wins when the encoder/global stack is
the expensive part. BLT-D/DV cut the decoder side instead.

### Speculative acceptance

![acceptance](docs/acceptance.png)

BLT-S on the well-trained plain model accepts 82% of drafted bytes. On
the BLT-D model it drops to 61% - the diffusion objective costs the
causal head some precision, and the tax reappears at inference.
BLT-DV acceptance (15-43%) is limited by the fully-masked regime being
under-trained at this scale.

### Wall-clock

![latency](docs/latency.png)

Plain BLT-S k=16 is 5.1x faster than greedy end-to-end. Raw BLT-D B=16
is 13x faster but quality-divergent. (Note: the BLT-S path uses the
KV cache, the greedy baseline does not - treat as directional.)

### A correctness detail worth knowing

Verified generation (BLT-DV) equals greedy output byte-for-byte **only
when commit points land on patch boundaries**. The verifier now forces
a patch boundary at each commit point; with content-driven patchers the
equality is approximate (agreement 0.59-1.00 measured) and the
benchmark gates it at >= 0.90. Unit tests enforce the exact guarantee
under controlled patching.

**Full numbers, the complete results table, and three tested-and-
rejected improvement ideas: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).**

Reproduce:

```bash
make bench-infer                       # needs runs/*_40k.fblt checkpoints
source .venv/bin/activate
python3 tools/bench_plots.py    # writes runs/bench_graphs/
```

## TODO

**Done:**
- Tensor and memory system (arena allocator, shapes, strides)
- Backend dispatch (CPU/CUDA abstraction)
- Core ops on CPU (elementwise, matmul, softmax, attention, etc.)
- Entropy model, patcher
- Attention (causal, cross-attention, custom masks)
- Full test harness (unit, integration, golden files)
- Hash n-gram embeddings
- Local encoder + decoder
- Patch transformer (global model)
- End-to-end forward pass
- FLOP counting and profiling
- Ablation sweeps (BPB tables)

**Todo:**
- CUDA kernels (validate 1:1 against CPU ground truth)
- Multi-GPU / larger-scale training runs
- Inference improvements from the benchmark findings (see
  `docs/BENCHMARKS.md`)

---

## Docs

- `API_REFERENCE.md` - Tensor and op API
- `docs/BENCHMARKS.md` - Full benchmark report (training trade-offs,
  inference methods, correctness analysis)
- `TEST_API_REFERENCE.md` - Writing tests