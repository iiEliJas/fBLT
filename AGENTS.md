# AGENTS.md

FBLT: byte-level language model (Meta's BLT/FBLT papers) in pure C99 + CUDA. CPU and CUDA are both complete: ops, models, training loop, and all three Fast-BLT inference modes (BLT-S / BLT-D / BLT-DV). 72/72 tests pass on both backends. Full CLI reference in `docs/CLI_REFERENCE.md`.

## Build & test

- `cmake --build build --target test` — build + run the entire suite (~4s). This is what CI runs (`.github/workflows/ci.yml`); there is no separate lint/typecheck.
- No test framework and no CLI filter for single tests. Suite selection is hardcoded in `tests/test_main.c`: `int do_test[4] = {core, model, parity, tools}`. To iterate on one area, flip that suite's flag, rebuild, and revert before committing. Note the `tools` suite ships disabled (`do_test[3] = 0`).
- Tests must run from the repo root: parity tests load golden files via relative paths (`data/*.bin`). The `test` custom target handles this; running `build/test_main` from elsewhere fails.
- Other targets: `cmake --build build --target main` (builds `build/main`), `cmake --build build --target sandbox_run` (builds + runs scratch playground), `cmake --build build --target bench-infer`, `cmake --build build --target train_sweep`.
- Toolchain: `-std=c99 -Wall -Wextra -D_POSIX_C_SOURCE=200809L -lm` comes from CMakeLists.txt; override with e.g. `cmake .. -DCMAKE_C_COMPILER=clang -DCMAKE_C_FLAGS="-O3 -std=c99 -Wall"`.
- CUDA backend (Linux + NVIDIA GPU): `cmake .. -DUSE_CUDA=ON` enables CUDA support; compiles `.cu` files with nvcc and links cuBLAS/cudart; outputs go to `build-cuda/`. `cmake --build build-cuda --target cuda-smoke` runs the device sanity check. CMakeLists auto-prefers `/usr/local/cuda/bin/nvcc` when a toolkit is installed there (see the CUDA Setup block); explicit `-DCUDAToolkit_ROOT=...` / `-DCMAKE_CUDA_COMPILER=...` still win. Do not use the apt `nvcc` package — its cuBLAS 12 runtime breaks against newer drivers. Toggle modes freely, but never reuse objects across them.
- WSL2 CUDA runtime: if a Linux NVIDIA driver was installed via apt alongside WSL2's own GPU passthrough, the apt `libnvidia-ptxjitcompiler.so.1` (old version in `/lib/x86_64-linux-gnu`) shadows the WSL2 driver's matching JIT and binaries crash in `__cuda_CallJitEntryPoint`. Workaround: put the WSL2 driver dir first on `LD_LIBRARY_PATH`, e.g. `LD_LIBRARY_PATH=/usr/lib/wsl/drivers/nvmdsi.inf_amd64_*/:/usr/lib/wsl/lib:/usr/local/cuda/lib64`. Also `CUDA_VISIBLE_DEVICES=0` is needed if the shell exports an empty `CUDA_VISIBLE_DEVICES=''`.
- CUDA profiling: `cmake --build build-cuda --target train_blt_d && nsys profile ...` (see `docs/PROFILING.md`).
- Clean: `rm -rf build/` (CPU) or `rm -rf build-cuda/` (CUDA).

## Training executable: `train_blt_d` (BLT-D diffusion training)

**Binary**: `build/train_blt_d` (CPU) or `build-cuda/train_blt_d` (CUDA)

**Core args**:
- `--corpus FILE` (required) — raw byte corpus
- `--steps N` (default 2000) — optimizer steps
- `--lr F` (default 0.05) — learning rate (SGD with global norm clip 5.0)
- `--block-size B` (default 4) — diffusion block size
- `--window W` (default 48) — clean-sequence length per example
- `--embed E` (default 64) — model width
- `--hidden H` (default 128) — FFN width
- `--layers L` (default 2) — layers per submodule (encoder/global/decoder)
- `--d0-mode MODE` (default learned) — `zeros` | `learned` for decoder d0 embed
- `--seed S` (default 7) — RNG seed
- `--report-every K` (default 25) — print frequency
- `--deterministic` (default off) — deterministic training (slower, no cuda atomics)

**Optimizer**:
- `--optimizer sgd|adamw` (default sgd) — optimizer selection
- `--beta1 F` (default 0.9) — AdamW first moment decay
- `--beta2 F` (default 0.999) — AdamW second moment decay
- `--eps F` (default 1e-8) — AdamW numerical stability
- `--weight-decay F` (default 0.01) — AdamW decoupled L2 weight decay

AdamW uses the standard PyTorch-compatible algorithm (bias-corrected m/v). SGD uses vanilla gradient descent with global-norm clip at 5.0.

**Mode selection**:
- `--diffusion 0|1` (default 1) — 0 = plain causal BLT baseline, 1 = BLT-D (masked diffusion)

**Checkpointing**:
- `--save-weights PATH` — write weights after training
- `--load-weights PATH` — load weights before training (steps=0 → eval only)
- `--eval-corpus FILE` — held-out corpus for causal BPB eval
- `--eval-windows N` (default 200) — eval window count
- `--eval-skip N` — bytes to skip before first eval window

**Diffusion schedule (stabilization)**:
- `--t-min F` (default 0.1) — diffusion timestep floor (clamps `t ~ U(0,1)`); higher = more stable gradients
- `--lr-decay 0|1` (default 0) — ×0.3 at 60% and 85% of steps
- `--mask-warmup N` (default 0) — ramp L_mask scale 0→1 over N steps
- `--mask-scale F` (default 1.0) — ceiling for L_mask weight (1.0 = paper Eq. 7)
- `--mask-late-step N` + `--mask-late-scale F` — raise cap from mask_scale to mask_late_scale between this step and end (paper §6 reweighting; 0 = disabled)
- `--t-warmup-hi F` + `--t-hi-start F` — high-t curriculum: fraction of steps over which t floor decays from t_hi_start down to t_min (0 = disabled)
- `--last-row-scale F` (default 1.0) — scale on row N-1's L_clean under corrected window+1 targets (`targets != NULL`): the row counts w times (loss adds `(w-1)/N * CE_last`, backward scales that row's gradient by w) while interior rows keep 1/N; ~511 evens its gradient with the interior rows. Inert when `targets == NULL`. 1.0 = off

**Patching**:
- `--entropy-patches` — segment training/eval windows with entropy LM + patcher (matches inference)
- `--train-entropy-lm PATH` — train standalone entropy LM (plain CE) and save to PATH
- `--entropy-lm PATH` — load pretrained entropy-LM weights for `--entropy-patches` segmentation

**Backend**:
- `--backend cpu|cuda` — select backend (CUDA requires `cmake .. -DUSE_CUDA=ON` build)

**Per-submodule layer overrides**:
- `--enc-layers N` — encoder layers (default `--layers`)
- `--glob-layers N` — global transformer layers
- `--dec-layers N` — decoder layers
- `--cross-attn all|last` (default all) — cross-attention placement: all layers or only final local layer

## CMake targets

| Target | Binary | Description |
|--------|--------|-------------|
| `cmake --build build --target test` | `build/test_main` / `build-cuda/test_main` | Build + run full test suite (72 tests) |
| `cmake --build build --target main` | `build/main` / `build-cuda/main` | Minimal main (empty, for linking) |
| `cmake --build build --target train_blt_d` | `build/train_blt_d` / `build-cuda/train_blt_d` | **Main BLT-D trainer** (see args above) |
| `cmake --build build-cuda --target cuda_bench` | `build-cuda/cuda_bench` | CUDA/CPU micro/meso/macro benchmark → `bench/results.jsonl` |
| `cmake --build build --target bench-infer` | `build/infer_bench` | Inference throughput benchmark (greedy, speculative) |
| `cmake --build build --target train_sweep` | `build/train_sweep` | Ablation sweep runner (config-driven) |
| `cmake --build build --target sandbox_run` | `build/sandbox` | Scratch playground |
| `cmake --build build-cuda --target cuda-smoke` | `build-cuda/cuda_smoke` | Device sanity check (H2D → kernel → D2H) |
| `cmake --build build-cuda --target cuda-sanitize` | — | Run tests under `compute-sanitizer` (native Linux only) |

CUDA builds: `cmake .. -DUSE_CUDA=ON` (outputs go to `build-cuda/`).

## Adding files and tests

- CMakeLists.txt source lists are explicit: new files under `src/` or `tools/` must be added to `BLT_CORE_SOURCES` by hand. Test sources are wildcarded only as `tests/unit/*/*.c` and `tests/parity/*.c` (one directory level — a new subdirectory layout won't be picked up).
- New test = function `int run_...(void)` returning 1 pass / 0 fail, declared in `tests/test_suite.h`, registered in a suite array in `tests/test_main.c`. Use `TEST_ASSERT` / `TEST_ASSERT_CLOSE` from `tests/test_helpers.h` (they return 0 on failure, so no cleanup after them).

## Parity / golden-file workflow

- `data/` holds binary golden tensors: uint32 ndim + uint32 dims header, then float32 payload (little-endian). They are generated only by the PyTorch/numpy reference scripts in `tests/py/` (e.g. `python tests/py/parity_patcher.py`, writes to `data/`); the C tests only consume them.
- Changing an op's intended numerics means regenerating goldens from `tests/py/` and committing both sides together. Unexplained golden diffs mean your op broke.

## Architecture rules

- One interface, two backends: signatures live in headers under `include/blt/{ops,models}/`, CPU implementations in `src/backend_cpu/`, dispatch through `src/core/backend.c`. Model code must stay backend-agnostic — don't add backend branches outside the dispatch layer. CUDA kernels go 1:1 at `src/backend_cuda/<same filename>.cu` (kernel files are `.cu`; headers stay backend-agnostic `.h` with `extern "C"` guards) and must be validated against CPU output.
- CPU is ground truth: every op gets a working C implementation first; GPU versions are checked against it.
- Ops allocate intermediates from the caller-passed `blt_arena` (bump allocator) — no malloc/free inside ops; callers own arena reset strategy (per-layer vs whole-forward matters for backward passes). Validation failures use `BLT_FATAL` (prints to stderr and exits), not error codes.
- Config-driven ablation options live in structs (`blt_transformer_config`: norm/activation type, cross-attention placement, k-split). JSON config plumbing does not exist yet despite README mentions of `configs/`.
- Inference code lives in `src/infer/` (block_generation, kv_cache, rope_gather, self_speculation, stats). The `tools/` directory has `generate_greedy.c` and `flops.c`.

## Working state notes

- `docs/HANDOFF_RESET.md` documents a previous CUDA-graph-capture branch that was archived. It contains important context on common misattributions (per-step loss noise from random timestep draws is normal; CPU vs CUDA loss divergence over many steps is expected numerical drift, not a bug).
- Ablation sweeps use fixed-stride patching during training (entropy patching at inference only). This is relevant for CUDA graph capture correctness.
- The fused-RMSNorm-weight gradient zeroing fix (8 gradient tensor types across encoder/global/decoder) is applied in `run/train_blt_d.c:830-843`. Forgetting to zero these is a common regression.

## Docs

- `docs/API_REFERENCE.md` — tensor/op API
- `docs/TEST_API_REFERENCE.md` — writing tests
- `docs/BENCHMARKS.md` — full benchmark report
- `docs/ABLATIONS.md` — ablation sweep results
- `docs/fast_bltPAPER.md` — paper on the fast BLT inference algorithm (drafts + verification)
- `docs/bltransformerPAPER.md` — paper on the BLT architecture (decoder + patcher)

## General Project Rules

### Branching Workflow
Never commit directly to `main`. Always create a feature branch before making changes. After completing each step in a plan, commit to that branch. When the work is done, summarize the changes and discuss with the user whether to merge onto `main`.

### Commenting Style
You are strictly forbidden from using block comments (`/* ... */`). All comments, including multi-line comments and JSDoc/docstrings, MUST use single-line comment syntax (`//`). Do not comment on any project plans or ideas in the codebase. All comments must be strictly technical and related to the code they are commenting on. Do not comment the file name at the start of a file. For example, do not write `// src/models/block_diffusion.c` at the top of a file.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
