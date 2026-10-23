# Comparison with facebookresearch/blt

This document compares fBLT with Meta's public Byte Latent Transformer repository:
<https://github.com/facebookresearch/blt>. The comparison is architectural and scope-based.
It is not a performance comparison: the projects target different scales, hardware, and
research questions.

## Scope

| | facebookresearch/blt | fBLT |
|---|---|---|
| Paper | Byte Latent Transformer (original BLT) | BLT, plus Fast-BLT extensions |
| Language/runtime | Python, PyTorch, xFormers | C and CUDA |
| Target scale | 1B and 7B models | 7.8M parameters demonstrated |
| Training | FSDP, BF16, torchrun, large corpora | Single GPU/process, FP32, small corpora |
| Inference | Autoregressive generation | Greedy, sampling, BLT-S, BLT-D, BLT-DV |
| Data | FineWeb/DCLM-style corpora | Source-code sample corpus and TinyStories |
| Released weights | 1B and 7B Hugging Face checkpoints | 7.8M checkpoint and entropy LM |
| License | CC-BY-NC-4.0 | Apache 2.0 |

## Shared ideas

Both projects implement the core BLT idea:

- Byte-level modeling without a fixed vocabulary.
- Entropy-based dynamic patching.
- A local encoder that pools bytes into patch representations.
- A global transformer over patches.
- A local decoder that expands patches back to bytes.
- Cross-attention between local and global representations.
- Hash n-gram embeddings for byte patterns.

## Where fBLT differs

### Implementation language

fBLT is a compact C/CUDA implementation with explicit arenas, CPU/CUDA backends, and a C test
suite. The official repository is a PyTorch research codebase built on Meta Lingua and
xFormers.

### Fast-BLT extensions

fBLT implements Fast-BLT inference modes that are not present in the public BLT repository:

- **BLT-S (self-speculation):** draft bytes decoder-only, then verify with one full forward.
- **BLT-D (block diffusion):** generate multiple bytes in parallel from masked states.
- **BLT-DV (diffusion + verification):** BLT-D drafting plus verification, producing greedy-identical output.

The public BLT repository provides standard autoregressive generation.

### Scale and validation

The official repository targets 1B-7B models on large web corpora. fBLT's demonstrated results
are at 7.8M parameters on TinyStories. fBLT's conclusions about BLT-DV are therefore
small-scale observations and should not be treated as evidence about the behavior of 1B-7B
models.

### Build and usability

fBLT is designed to build and run on a laptop or single GPU without PyTorch, CUDA Python
packages, or Hugging Face weights. The official repository requires a heavier Python/CUDA
environment and is oriented toward reproducing the paper's large-scale experiments.

### Documentation and reproducibility

fBLT includes a starter guide, CLI reference, research guide, experiment log, experiment
template, benchmarks, and ablations. The official repository is research code with example
configs and is still being updated for reproducibility.

## What fBLT does not implement from the official BLT stack

- Large-scale distributed training (FSDP, tensor parallelism, multi-node).
- BF16/FP8 mixed precision.
- Hugging Face checkpoint loading and `transformers` integration.
- The official data pipeline for FineWeb/DCLM.
- The official tokenizer/BPE hybrid patching modes in the same form.
- The full set of official model configurations and scaling experiments.

## What the public BLT repository does not implement

- Fast-BLT block-diffusion decoding.
- BLT-S self-speculation.
- BLT-DV verification.
- C/CUDA implementation or CPU-only execution.
- fBLT's diagnostic tools and leak monitor.

## Interpretation

fBLT is not a drop-in replacement for the official BLT codebase. It is a small, inspectable
implementation of BLT plus Fast-BLT inference experiments. Its value is in making the
mechanisms runnable, testable, and measurable at small scale, not in reproducing the paper's
largest models.

For implementation details, see [README.md](../README.md), [STARTER_GUIDE.md](STARTER_GUIDE.md),
and [BENCHMARKS.md](BENCHMARKS.md).
