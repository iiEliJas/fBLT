# E009: Backward-pass profiling and optimization targets

## Question
Where does training time go in `train_blt_d`, and what should be optimized first?

## Setup
- Config: research config (window 64, embed 64, hidden 128, layers 2, block size 4, batch 1 window).
- Tooling: `BLT_TRAIN_TIMING=1` for CPU/CUDA component breakdown; `nsys` for kernel-level view.
- Backends: CPU and CUDA.

## Results

| Backend | ms/step | fwd | bwd | opt |
|---------|---------|-----|-----|-----|
| CPU FP32 | 50.9 | 15.1 | 33.7 | 1.6 |
| CUDA FP32 | 26.2 | 4.8-7.8 | 12.9 | 4.9 |
| CUDA BF16 | 25.1 | 4.7-6.6 | 12.4 | 4.8 |

- Backward pass is ~50% of step time on both CUDA variants.
- Forward is already small (~5-8 ms).
- Optimizer ~5 ms.

## nsys findings
Top kernels by time:
- `ampere_sgemm_*` (many matmuls in backward)
- `blt_patch_pool_backward_kernel`
- `blt_ce_row_loss_kernel`
- `blt_indexed_row_accumulate_kernel`
- `blt_embedding_lookup_kernel`
- Large number of `[CUDA memset]` (5731 calls, 5.5 ms) and memcpy operations.

## Conclusion
Backward matmuls and memory traffic (memset/memcpy) dominate. Future speed work should prioritize:
1. Reducing temporary allocations / memset traffic in backward paths.
2. Fusing backward elementwise kernels where possible.
3. FP8/quantized matmuls for backward GEMMs.

## Status
Checked (exploratory profiling).

## Optimization results (feature/cuda-backward-opt)

| Change | ms/step (FP32) | bwd | opt |
|--------|----------------|-----|-----|
| Baseline | 26.2 | 12.9 | 4.9 |
| + async memset + skip redundant grad passes (grad_accum==1) | 22.1 | 12.6 | 4.3 |
| + batched gradient-norm download in clip_all | 19.9 | 12.0 | 1.0 |
| + skip zero-init for fully-overwritten upload tensors | 20.2 | 12.6 | 1.1 |

Net: ~23% faster (26.2 -> 20.2), loss unchanged (556.8496).

## Remaining (item 26)

- 26.4 (fold clip scale into optimizer): low value; clip scale only runs when
  grad norm exceeds max_norm, rare in practice.
- 26.6 (fuse elementwise chains): moderate effort; would reduce the ~197
  `blt_add_kernel` launches per step.
- 26.7 (CUDA graph capture): highest potential win, but the step mixes host-side
  per-step variation (timestep draw, patch count, text) with device kernels, and
  several kernels read host-side structs by pointer. A correct capture needs
  those inputs moved to device buffers updated before replay. Deferred to a
  dedicated branch.
