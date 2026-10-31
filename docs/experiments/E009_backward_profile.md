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
