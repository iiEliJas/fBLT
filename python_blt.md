# Fast PyTorch BLT / BLT-D: Analysis and Plan (planning only)

## Summary

This plan proposes a native PyTorch implementation of BLT / BLT-D (plus BLT-S / BLT-DV inference) that reuses the official `facebookresearch/blt` code only as reference (license) and re-implements Fast-BLT-only parts from the C reference in `csrc/`. The C code remains ground truth; Python must match logits/losses on shared golden fixtures. Speed is first-class: bf16 autocast, FlexAttention/SDPA block masks, `torch.compile`, batched multi-window steps, pinned memory, CUDA graphs where shapes are static.

Key facts:
- Official repo `~/projects/blt` is licensed **CC BY-NC 4.0**; cannot be vendored into this Apache-2.0 repo. Re-implement with attribution.
- C entry points: `csrc/train_blt_d.c`, `csrc/infer.c`, `csrc/models/block_diffusion.c`.
- Existing `fblt/` is only a wrapper around C binaries.
- Checkpoint format: `FBLT` magic, version 2, JSON meta, then tensors (`csrc/models/checkpoint.c`).

## Component mapping table

| Component | Paper | C code | Official repo | Reuse vs new | Notes |
|---|---|---|---|---|---|
| Entropy patcher | BLT §2.3 | `csrc/models/patcher.c` `blt_segment_patches` (59-99); `entropy_lm.c` (58-105) | `bytelatent/data/patcher.py` | Ideas only | C default fixed-stride-4 in training; entropy patching optional |
| Entropy LM (.fblt) | BLT §4.2 | `entropy_lm.c`; checkpoint v1 (215-291); `model_builder.c` `blt_make_entropy_lm` (61-89) | `transformer.py` `LMTransformer` | New loader (interop) | 256 vocab, embed 32, 1 layer, 2 heads, hidden 64, rope theta 10000 |
| Byte embedding | BLT §3.2 | `byte_embedding.c` (16-88); vocab 256/257 w/ mask | `model/blt.py` embeddings | Reimplement | Decoder vocab 256 + `BLT_MASK_TOKEN_ID=256` |
| Hash n-gram | BLT §3.2.1 | `hash_ngram.c` (16-264); defaults `model_builder.c:30-33` | `model/blt.py` rolling hash | Reimplement (C ground truth) | sizes {3,4}, 50 buckets, prime 1e9+7, normalize |
| Local encoder | BLT §3.2 | `local_encoder.c` (486) | `model/local_models.py` `LocalEncoder` | Reimplement/align | mean pool, cross-attn placement |
| Cross-attention | BLT §3.2.2/3.3.1 | `cross_attention.c` (353); `mask_builder_cpu.c` | `latent_transformer.py` `CrossAttention`; FlexAttention | Reuse FlexAttention idea | additive masks in C |
| Global transformer | BLT §3.1 | `global_transformer.c` (140) | `latent_transformer.py` | Reimplement | pre-norm |
| Local decoder (clean) | BLT §3.3 | `local_decoder.c` (543) | `local_models.py` `LocalDecoder` | Reimplement | LM head [E,256] |
| BLT-D block construction + masks | Fast-BLT §3.2.1/3.2.2 | `block_diffusion.c` `blt_block_batch_build_t`; `diff_ctx_build`; mask builder | None | New | Figure-5 mask |
| Losses | Fast-BLT §3.2.3 | `block_diffusion.c` forward; `train_blt_d.c:703-720` | None | New | paper-loss sums, mask_loss_norm, t_min, warmup, late scaling, last_row_scale |
| Unmasking | Fast-BLT §3.1.2 | `block_generation.c` `blt_unmask_select_confidence` (20), `_eb` (32), `blt_draft_block` (156) | None | New | conf threshold 0.7, EB 1.0 nats |
| BLT-S | Fast-BLT §5.1 | `self_speculation.c` `blt_generate_greedy_selfspec` (336); `kv_cache.c` | None | New | two-tier KV cache |
| BLT-DV | Fast-BLT §5.2 | `block_generation.c` `generate_common` (350) do_verify=1 (481); verify functions | None | New | adaptive B |
| Evaluation | Fast-BLT §4.2/6 | `train_eval.c`; `train_blt_d.c:1000-1210`; shadowed `final_masked_acc` at 1169 | `eval.py` | Reimplement | leak tripwire monitor |
| Data pipeline | BLT §4.1 | `train_blt_d.c` windowing (670-710) | `data/*`, preprocess | Reuse packing | C uses 1 window/step; Python batches |
| Checkpointing | — | `checkpoint.c` (140-433); v2; config keys (1001,1098) | `checkpoint.py` | Interop | FBLT v2 |

## Discrepancies (paper / C / official)

1. Loss normalization: paper Eq.7 sums; C supports paper-loss (sums) and mask_loss_norm (mean over masked cells) plus last_row_scale. Follow C.
2. t floor: paper U(0,1); C clamps t_min (default 0.1) + high-t curriculum. Follow C.
3. Mask warmup/late scaling: C ramps L_mask 0->1 and raises cap late. Follow C.
4. d0 modes: zeros (default) and learned. Expose both, default zeros.
5. Cross-attn placement: all vs last. Default all.
6. Training patching: C fixed-stride-4 default; entropy patching optional. Follow C.
7. Decoder length: S = N + (M-1)*B. Match indexing.
8. C bug: shadowed `final_masked_acc` at line 1169. Python implements intended outer behavior and documents difference.
9. RNG: C splitmix64. Python injectable torch.Generator; reference reproduces where feasible.
10. Precision: C fp32; Python bf16 training. Parity in fp32 reference mode.

## Performance plan and benchmarks

Baseline: build C trainer, run `BLT_TRAIN_TIMING=1`, log t_fwd/t_bwd/t_opt/t_step (`train_blt_d.c:658-660`).

Targets (RTX 4060 8GB): >=10x step throughput vs C; GPU util >=70%; peak mem <=6GB for 5.8M params with 8-16 windows; no .item()/sync in hot step except logging.

Harness: `bench/py_bench_train.py` using CUDA events; report steps/s, bytes/s, peak VRAM, per-module ms, compile warmup; store `bench/results_python.jsonl`.

Profiling: torch.profiler traces; nsys; torch.cuda.memory_summary.

Optimizations: bf16 autocast (loss/softmax fp32); FlexAttention BlockMask for encoder/cross/Figure-5; SDPA for static masks; torch.compile reduce-overhead; batch multiple windows; DataLoader workers + pin memory; bucket by (N,M,S) + padding; CUDA graphs for fixed buckets; vectorize mask/loss; cached prefix latents/KV for inference.

Numerical parity: fp32 eager reference vs C within 1e-4 logits / 1e-5 loss; optimized bf16 within 1e-2.

## Config and shared-test design

Config: YAML/TOML both sides. C parses documented subset. Existing `fblt/_config.py` uses YAML; align keys with C flags and defaults (`model_builder.c`).

Shared tests: reuse `tests/py/golden.py` writer and `data/*.bin` format. Add goldens for patcher, encoder, global, decoder clean/corrupt, mask, L_clean/L_mask, inference modes. RNG injection for parity.

Weight interop: implement `FBLT` v2 reader/writer. Header magic/version/meta/num_tensors; tensor records u16 name, u32 ndim, size_t dims, f32 payload. Names/order per `checkpoint.c:9-140`. Entropy LM v1. Meta includes config keys. Export fp32 from Python.

## Package architecture

Monorepo under `pysrc/`:
```
pysrc/fblt_torch/...
```
Keep existing `fblt/` CLI wrapper; add native backend selection. Entry points `fblt-train`, `fblt-infer` dispatch C or Python. Logging same JSONL schema as C.

## Milestones with acceptance tests

- M0: tooling/skeleton; pytest runs.
- M1: models fp32 eager; encoder/global/clean decoder match C goldens 1e-4.
- M2: hash n-gram + cross-attention goldens.
- M3: BLT-D block construction + masks match C for fixed seed.
- M4: BLT-D loss matches C for same batch (fp32, injected t/mask).
- M5: backward + SGD update match C within 1e-3.
- M6: checkpoint interop both directions.
- M7: throughput >=10x C; bf16+compile.
- M8: inference BLT-S/D/DV match C greedy sequences.
- M9: eval metrics match C.
- M10: tiny overfit regression + checkpoint round-trip.

## Risks and unknowns

- FlexAttention constraints for Figure-5 dynamic masks; bucketing/SDPA fallback.
- Shape dynamism from entropy patching; fixed-stride training.
- Memory for S=N+(M-1)B; gradient checkpointing.
- Numerical tolerance fp32 vs bf16.
- RNG parity; inject seeds.
- Leak tripwire C bug (line 1169) handled intentionally.
- License: no vendoring official code.
- Unknown C internals (e.g. `blt_patch_decoder_latent_at`) to read during implementation.

## Decisions (final)

Approved conventions:
- Functional core with thin modules; composition only (BLT and BLTD share encoder+global by composition, separate classes); frozen nested dataclass configs; PatchLayout built outside forward; explicit inference state objects; plain explicit training loop; paper/C names; type hints and shape comments; `*_ref` colocated with identical signatures and equivalence tests; FlexAttention BlockMask as production default; uv, ruff, pyright (standard mode), pytest; keep existing `fblt/` wrapper untouched until parity milestone passes, then add backend selection last; `pysrc/` layout.

Approved changes:
1. Module name `global.py` is a keyword: use `models/global_model.py`. Add `models/entropy_lm.py`.
2. Package name is `fblt_torch` (`pysrc/fblt_torch/`).
3. Everything batched: tensors `(bsz, ...)`. Never bare `B`; use `block_size` for diffusion block and `bsz` for batch. Window length N fixed per run. Patch dimension padded to bucket sizes with validity masks.
4. `forward` returns logits only (small dataclass with clean-row and block-row logits). Losses are separate pure functions (`l_clean`, `l_mask`). A `training_step` function composes model + losses and is what gets compiled.
5. bf16 autocast, fp32 master weights and optimizer state, fused AdamW, no GradScaler. Grad accumulation yes. TF32 off and eager mode in all parity tests.
6. Unmasking functions take a `still_masked` mask, operate on device without Python control flow, implement "always unmask at least one position", support top-p sampling with injectable Generator. Track NFEs, acceptance rate, memory-bandwidth formula (Eq. 8).
7. Licensing: write original code from papers and C code. No verbatim copying, no vendoring, no line-by-line translation of `~/projects/blt/`. Use that repo only to resolve ambiguities and as optional test oracle. Quote both license statements verbatim in plan and state repo license. Provenance notes in `python_blt.md` and module docstrings.
8. Parity tests inject exact masks and t dumped by C. Do not port C RNG. Separately test random sampling path statistically.
9. Docstrings cite file and function names of C code or paper equation/section. File:line references live only in `python_blt.md`.
10. Enforceable speed rules, as tests: training step runs with `torch.cuda.set_sync_debug_mode("error")`, compiles with `fullgraph=True`, benchmark with recorded number before optimization merged.

Defaults:
- Config: TOML (`tomllib`, Python >= 3.11). C parses documented minimal subset. Config keys match C flag names. Implement every C loss/schedule option or list unsupported with reasons.
- Layout: monorepo. `configs/` at top level; `tests/shared/` holds cross-implementation fixtures (raw little-endian tensors + JSON manifest). Python-only tests in `pysrc/tests/`. Parity runs both directions.
- Checkpoints: read/write C fblt format for interchange; separate torch checkpoint (weights, optimizer, scheduler, RNG) for resume.
- Compile policy: default mode first, `dynamic=False` with bucketed shapes; CUDA graphs and max-autotune only as separately benchmarked options; compile can always be switched off in config.
- Data: memory-mapped corpus; entropy patch boundaries precomputed once on GPU in large batches and stored as memmapped array keyed by hash of (corpus, entropy LM, window, patcher config). Provide function reproducing C's `train_window_index` for tests.
- Logging: JSONL metrics with same event schema as C metrics log, plus one stdout line every `report_every` steps. Log raw loss as in C and per-byte bits (bits/byte). No external logging backend for now.
- Multi-GPU: single GPU first. Keep DDP-ready (no global state, rank-aware logging and seeding), FSDP out of scope.
- Type checking: comments for shapes, no jaxtyping runtime checks. Shape asserts only in `*_ref` functions and tests.

## Style proposals (conventions)

### Architecture
- Functional core, thin modules; composition only.
- Frozen dataclass configs, nested per submodule.
- PatchLayout built outside forward.
- On-device RNG with injectable `torch.Generator`.
- Inference uses explicit state objects.
- Plain explicit training loop.

### Naming and documentation
- Paper/C names; checkpoint tensor names mirror C format.
- Docstrings cite paper equation/section or C function.
- Type hints everywhere; shape comments.

### Speed and testing
- Every optimized op has `*_ref` with identical signature and equivalence test.
- No optimization merged without benchmark number and parity test.
- Tooling: uv, ruff, pyright, pytest.

### Module layout (pysrc/)
```
pysrc/
  fblt_torch/
    __init__.py
    config.py
    checkpoint.py
    patching.py
    layouts.py
    masks.py
    losses.py
    unmask.py
    verify.py
    rng.py
    models/
      byte_embedding.py
      hash_ngram.py
      encoder.py
      global_model.py
      decoder.py
      entropy_lm.py
      blt.py
      bltd.py
    infer/
      self_spec.py
      block_diff.py
      block_dv.py
      kv_cache.py
      state.py
      generate.py
    data/
      dataset.py
      collate.py
      patch_layout_gpu.py
    train.py
    eval.py
    bench.py
```

### Main signatures (prose stubs)
- `build_patch_layout(...) -> PatchLayout`
- `figure5_mask(...) -> Tensor[S,S]` or BlockMask
- `l_clean(...) -> scalar`
- `l_mask(...) -> scalar`
- `unmask_confidence(...) -> (B,) bool`
- `unmask_eb(...) -> (B,) bool`
- `verify_candidate(...) -> (accepted_len, corrected_bytes)`
- Top-level: `BLT.forward(...)`, `BLTD.forward(...)`.

### Points needing your decision
- PatchLayout: dataclass vs NamedTuple (recommend dataclass).
- Mask construction default: additive vs BlockMask with SDPA fallback (recommend BlockMask).
- `*_ref` location: colocate with `_ref` suffix vs `ref/` package (recommend colocate).
- One class vs mode flag (recommend separate classes).
- Training utilities: include grad accumulation + AMP scaler (recommend yes).
- Keep existing `fblt/` wrapper or replace entry points (recommend keep wrapper, add native backend selection).

## Licensing details for `~/projects/blt`

- License: `~/projects/blt/LICENSE` = **CC BY-NC 4.0**; `bytelatent/model/blt.py` declares `license_name="fair-noncommercial-research-license"`.
- Copying code verbatim into Apache-2.0 repo is not allowed for non-NC use; even research requires attribution and cannot be relicensed.
- Re-implementing ideas/algorithms is allowed; attribute paper + repo.
- Practical: treat official repo as reference only; write original code; document provenance; keep NC-only snippets in a separate, clearly marked module if used.

## M11+ plan: complete 1:1 fast BLT in Python

### Current status (2026-10-29)
- M1–M7, M9, M10: validated against C goldens (within tolerance).
- M8: functional (infer decoder + block draft + generation smoke), not parity-validated.
- Entry points: not wired to native Python.
- Export conversion: not implemented (torch `[out,in]` ↔ C `[in,out]`).
- BLT-S / BLT-DV: not implemented.
- Performance: CPU benchmark only.

### Remaining work

#### 1. Entry-point wiring (`fblt/train.py`, `fblt/infer.py`)
- Add `--backend torch` to both CLIs (currently only `cpu|cuda`).
- If backend is torch, dispatch to native Python instead of resolving a C binary.
- Native training entry point: `pysrc/fblt_torch/cli.py` (train/infer subcommands) reading the same YAML config.
- Keep C path unchanged; default backend remains cpu for compatibility.

#### 2. Native training loop (BLT-D)
- Implement full training step: data loading (memory-mapped corpus), patching (fixed/entropy), encoder+global+decoder, BLT-D loss, backward, AdamW, grad clip, schedules, checkpointing, metrics JSONL.
- Reuse `fblt_torch` modules and `losses.py`.
- Support all C options: paper-loss, mask_loss_norm, t_min, t-curriculum, mask warmup/late scale, last_row_scale, d0 mode, cross-attn mode, entropy patches, eval corpus, monitor.

#### 3. Model export conversion (torch ↔ C)
- Implement `export_c_checkpoint(model, path)` that transposes every linear weight to C layout `[in,out]`.
- Implement `load_c_checkpoint(path, model)` inverse.
- Byte-identical round-trip test on a trained model.

#### 4. Inference parity (M8)
- Port exact infer mask and group rules (already extracted in `patching.py`).
- Implement verification (`verify_candidate`) shared by BLT-DV and BLT-S.
- BLT-D: fixed/entropy patching, block drafting, unmasking.
- BLT-S: self-speculation with two-tier KV cache.
- BLT-DV: diffusion + verification with adaptive B.
- Compare greedy output to C on tiny prompts (temp=0).

#### 5. Entropy LM
- Train and run entropy LM in Python; read/write `.fblt` format.
- Precompute patch boundaries on GPU and cache.

#### 6. Performance
- bf16 autocast, fp32 master weights, fused AdamW.
- `torch.compile` on training step (fullgraph=True in tests).
- FlexAttention BlockMask production; SDPA fallback.
- Batched multi-window steps, pinned memory, bucketed shapes.
- CUDA graphs for static shapes.
- Benchmark and record numbers.

#### 7. Config completeness
- Implement every C loss/schedule option or list unsupported with reasons.
- TOML config parsed by both sides.

#### 8. Testing
- Unit, equivalence (`*_ref`), cross-implementation parity.
- `torch.cuda.set_sync_debug_mode("error")` in training step tests.
- No tolerance loosening.

### Decisions (final, unchanged)
All prior decisions stand: monorepo `pysrc/`, YAML/TOML config, uv/ruff/pyright, Pythonic names with C aliases, re-implementation with attribution, Apache-2.0, single GPU first, JSONL + wandb, no GitHub pushes until licensing complete.