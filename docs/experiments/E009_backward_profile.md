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

## CUDA graph capture attempt (feature/cuda-graph-train)

Attempted to capture the full diffusion training step as a CUDA graph to eliminate
launch overhead (GPU is ~15% busy; ~1190 launches/step at ~9.5 us/launch).

### Blockers found

1. **Synchronous memcpy/memset during capture.** `blt_cuda_memcpy_h2d` uses
   `cudaMemcpy` (sync) and `blt_cuda_memset` used `cudaMemset` (sync). CUDA does
   not allow synchronous copies during stream capture. Converting these to async
   versions is required throughout the capture path.
2. **Host-side scratch arena state.** `blt_embedding_lookup_cuda` allocates from
   `blt_cuda_get_scratch_arena()` and advances its offset on the host. This is
   not a CUDA API call but couples host state to capture; the arena must be reset
   to the same offset before every replay.
3. **Per-step varying inputs read on host.** `batch->t`, `batch->positions`,
   `batch->tokens` are read on the host during the forward pass and used to
   derive kernel arguments. In a captured graph those arguments are frozen at
   capture time. Making the graph correct requires moving these inputs into
   device buffers that kernels read at replay time, and refactoring the decoder
   to read from device memory instead of host structs.
4. **Loss download (D2H).** Must be moved outside the capture region.

### Conclusion

Full CUDA graph capture is feasible but requires a substantial refactor:
- async memcpy/memset everywhere in the capture path,
- device-resident per-step inputs (t, positions, tokens, text),
- decoder refactored to read inputs from device memory.

Estimated as a multi-week effort. Deferred to a dedicated branch. The simpler
optimizations already delivered ~23% speedup (26.2 -> 20.2 ms/step).

## Elementwise fusion (item 26.6, feature/elementwise-fusion)

Fused `blt_add(a,b,tmp) + blt_rmsnorm_backward(tmp,...)` into a single kernel
`blt_rmsnorm_backward_add` that sums the two gradient inputs internally,
eliminating one intermediate buffer + one kernel launch per call site.

Call sites updated: transformer_stack.c (grad_norm2_a + grad_norm2_b).

| Config | ms/step | fwd | bwd | opt |
|--------|---------|-----|-----|-----|
| Before (large config) | 46.5 | 12.8 | 30.0 | 2.3 |
| After  (large config) | 46.8 | 12.8 | 30.4 | 2.3 |

Net: within noise at this scale (single fusion, 4 calls/step). Deeper fusion
(chaining residual adds into attention/FFN backward) needed for measurable gain.
CPU ground truth parity maintained (80/80 tests pass).
