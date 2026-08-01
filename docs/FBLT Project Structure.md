
**Byte Latent Transformer + Fast-BLT — C/CUDA implementation** Target use case: a byte-level code-completion / small general LLM. No concept tier — this is plain BLT (entropy patcher → local encoder → global patch transformer → local decoder) plus Fast-BLT's inference-speed extensions, plus a dedicated phase for the architecture improvements the BLT paper's own ablations already point to.

---

## 1. Design principles behind the layout

1. **One interface, two backends.** Every op (matmul, attention, RMSNorm, ...) is declared once in a header and implemented once per backend (`backend_cpu/`, `backend_cuda/`). Calling code never knows which backend it's linked against.
2. **Phases add directories; they don't rewrite them.** The tree already contains folders for later-phase work that doesn't exist yet — you grow into it instead of restructuring into it.
3. **Ablation knobs are config, not code branches.** The BLT paper's own ablation section (§7) is a menu of hyperparameters (hash n-gram sizes/vocab, cross-attention placement, entropy-model size, local encoder/decoder depth split) — every one of these must be reachable from a config file, not hardcoded, so the improvement phase is "try new configs and benchmark," not "rewrite the model."
4. **Tests and benchmarks are siblings of the code**, present from the day a module is created, not appended at the end.

---

## 2. Top-level tree

```
blt/
├── CMakeLists.txt
├── cmake/
│   ├── FindCUDAToolkitOrSkip.cmake
│   └── Warnings.cmake
├── README.md
├── docs/
│   ├── phases/
│   │   ├── phase0_foundations.md
│   │   ├── phase1_entropy_model.md
│   │   ├── phase2_patcher.md
│   │   ├── phase3_local_encoder.md
│   │   ├── phase4_baseline_blt.md
│   │   ├── phase5_ablation_improvements.md
│   │   ├── phase6_fast_decoding.md
│   │   ├── phase7_cuda_scaling.md
│   │   └── phase8_data_eval.md
│   ├── architecture.md              # hierarchy diagram + design decisions, kept current
│   ├── ablations.md                 # running log: what you tried, config diff, benchmark result
│   └── metrics.md                   # definitions of every eval metric (BPB, NFEs, bandwidth, exact-match, ...)
│
├── include/blt/                     # PUBLIC headers — this is "the interface"
│   ├── core/
│   │   ├── tensor.h
│   │   ├── allocator.h
│   │   ├── backend.h                # enum {BLT_BACKEND_CPU, BLT_BACKEND_CUDA} + dispatch macros
│   │   └── dtype.h
│   ├── ops/
│   │   ├── matmul.h
│   │   ├── elementwise.h
│   │   ├── softmax.h
│   │   ├── rmsnorm.h
│   │   ├── rope.h
│   │   ├── attention.h              # generic masked MHA — reused at every tier
│   │   └── cross_attention.h        # bytes→patches pooling / patches→bytes unpooling
│   ├── model/
│   │   ├── entropy_model.h
│   │   ├── patcher.h
│   │   ├── hash_ngram.h             # rolling polynomial hash + embedding tables (BLT §3.2.1)
│   │   ├── local_encoder.h
│   │   ├── patch_transformer.h      # the "global latent transformer" of plain BLT
│   │   └── local_decoder.h
│   ├── data/
│   │   ├── byte_stream.h
│   │   ├── shard_format.h
│   │   └── code_corpus.h           # code-specific loading (repo dedup, language filter, license filter)
│   └── infer/
│       ├── kv_cache.h               # two tiers: byte-window cache, patch cache
│       ├── self_speculation.h       # BLT-S, Phase 6
│       ├── block_diffusion.h        # BLT-D, Phase 6, optional stretch
│       └── generate.h
│
├── src/
│   ├── core/
│   │   ├── tensor.c
│   │   ├── allocator.c
│   │   └── backend.c
│   │
│   ├── backend_cpu/                 # Phase 0 onward, always built
│   │   ├── matmul_cpu.c
│   │   ├── elementwise_cpu.c
│   │   ├── softmax_cpu.c
│   │   ├── rmsnorm_cpu.c
│   │   ├── rope_cpu.c
│   │   ├── attention_cpu.c
│   │   └── cross_attention_cpu.c
│   │
│   ├── backend_cuda/                 # grows from Phase 0/1 (boring ops) through Phase 7 (fused kernels)
│   │   ├── matmul_cuda.cu
│   │   ├── elementwise_cuda.cu
│   │   ├── softmax_cuda.cu
│   │   ├── rmsnorm_cuda.cu
│   │   ├── rope_cuda.cu
│   │   ├── attention_cuda.cu
│   │   ├── attention_varlen_cuda.cu  # Phase 7 only — fused variable-length kernel
│   │   └── cross_attention_cuda.cu
│   │
│   ├── model/                        # backend-agnostic: only calls through include/blt/ops/*
│   │   ├── entropy_model.c           # Phase 1
│   │   ├── patcher.c                 # Phase 2
│   │   ├── hash_ngram.c              # Phase 3
│   │   ├── local_encoder.c           # Phase 3
│   │   ├── patch_transformer.c       # Phase 4
│   │   └── local_decoder.c           # Phase 4
│   │
│   ├── data/
│   │   ├── byte_stream.c
│   │   ├── shard_writer.c
│   │   ├── shard_reader.c
│   │   └── code_corpus.c
│   │
│   └── infer/
│       ├── kv_cache.c
│       ├── self_speculation.c        # Phase 6
│       ├── block_diffusion.c         # Phase 6, optional stretch
│       └── generate.c
│
├── tools/
│   ├── dump_reference_io.py          # Phase 0 method: dumps random weights+IO for a given op
│   ├── preprocess_corpus.c           # runs entropy model + patcher, writes shards
│   ├── calibrate_threshold.c         # Phase 2: fits theta_g/theta_r for a target avg patch size
│   ├── train.c                       # training entrypoint, config-driven
│   ├── generate_cli.c                # inference entrypoint (completion-style: prompt -> continuation)
│   ├── flops_calculator.c            # Phase 4: BLT's own FLOPs equations (Appendix B), reused through Phase 7
│   ├── ablation_sweep.c              # Phase 5: runs a list of configs back-to-back, logs benchmark deltas
│   └── code_eval.c                   # Phase 8: exact-match / edit-distance / FIM-style completion scoring
│
├── tests/
│   ├── unit/                         # mirrors src/ 1:1
│   │   ├── core/
│   │   ├── backend_cpu/
│   │   ├── backend_cuda/
│   │   └── model/
│   ├── property/                     # patcher coverage, mask block-structure, hash collision sanity
│   ├── golden/
│   │   └── data/*.bin                # includes the BLT paper's own worked entropy example
│   ├── integration/
│   │   ├── test_phase1_entropy_overfit.c
│   │   └── test_phase4_blt_overfit.c
│   ├── parity/                       # CPU vs CUDA diff tests, one per ported op
│   └── test_main.c
│
├── bench/
│   ├── micro/                        # single-kernel timing
│   ├── meso/                         # one tier forward+backward
│   ├── macro/                        # full train step, full generation loop
│   ├── ablation/                     # Phase 5: paired before/after benchmark runs per config change
│   ├── profiler.h / profiler.c
│   └── report/                       # scripts rendering benchmark logs into tables/plots
│
├── configs/
│   ├── tiny_cpu_debug.json
│   ├── small_gpu.json
│   ├── target_1b.json
│   └── ablations/                    # one file per ablation variant, diffed against a baseline config
│       ├── baseline.json
│       ├── ngram_3_4_5_400k.json
│       ├── ngram_6_7_8_100k.json
│       ├── crossattn_decoder_only.json
│       ├── crossattn_encoder_pool_lastlayer.json
│       ├── entropy_model_50m_ctx512.json
│       ├── encoder1_decoder9.json
│       └── encoder5_decoder5.json
│
├── data/                              # gitignored
│   ├── raw/
│   └── shards/
│
├── third_party/
│
└── scripts/
    ├── run_all_tests.sh
    ├── run_all_benchmarks.sh
    ├── run_ablation_sweep.sh
    └── new_op.sh
```

---

## 3. The CPU/CUDA interface pattern, concretely

Every op follows the same three-file shape. Take `rmsnorm` as the running example:

**`include/blt/ops/rmsnorm.h`** — the only file calling code ever `#include`s:

```c
#ifndef BLT_OPS_RMSNORM_H
#define BLT_OPS_RMSNORM_H
#include "blt/core/tensor.h"

void blt_rmsnorm_forward(const blt_tensor* x, const blt_tensor* weight, blt_tensor* out);
void blt_rmsnorm_backward(const blt_tensor* grad_out, const blt_tensor* x,
                           const blt_tensor* weight, blt_tensor* grad_x, blt_tensor* grad_weight);
#endif
```

**`src/backend_cpu/rmsnorm_cpu.c`** implements `blt_rmsnorm_forward_cpu(...)` — plain C, no CUDA headers anywhere.

**`src/backend_cuda/rmsnorm_cuda.cu`** implements `blt_rmsnorm_forward_cuda(...)` — same signature, compiled only if `BLT_WITH_CUDA=ON`.

**`src/core/backend.c`** dispatches:

```c
void blt_rmsnorm_forward(const blt_tensor* x, const blt_tensor* w, blt_tensor* out) {
    if (x->backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        blt_rmsnorm_forward_cuda(x, w, out);
#else
        BLT_FATAL("built without CUDA support");
#endif
    } else {
        blt_rmsnorm_forward_cpu(x, w, out);
    }
}
```

Model code calls `blt_rmsnorm_forward(...)` and never mentions CPU or CUDA — a model built for CPU-only debugging and the eventual 1B CUDA run are **the same source file**. `BLT_WITH_CUDA` is a CMake option; when `OFF`, `src/backend_cuda/` isn't even compiled and `nvcc` isn't required.

---

## 4. Test tree mirrors source tree, 1:1

`tests/unit/backend_cpu/rmsnorm_cpu_test.c` sits next to `tests/unit/backend_cuda/rmsnorm_cuda_test.c`, both driven by the same golden input file in `tests/golden/data/rmsnorm_case1.bin`, produced once by `tools/dump_reference_io.py`. This is what makes the CPU-vs-CUDA parity check (`tests/parity/`) nearly free to write — same golden file, loaded twice, diffed.

|Test type|Lives in|Written when|
|---|---|---|
|Single op forward/backward vs. Python reference|`tests/unit/<mirrors src path>`|The op is written|
|Invariant/fuzz (masks, patcher coverage, hashing)|`tests/property/`|The module has a "shape" to check|
|Known input → known output, stored|`tests/golden/`|You've hand-verified an output once (e.g. the paper's worked entropy example)|
|Whole phase trains to near-zero loss on tiny data|`tests/integration/`|The phase's forward+backward is fully wired|
|Same op, CPU result vs. CUDA result|`tests/parity/`|An op gets a CUDA implementation|

---

## 5. Adding a new op — the five-minute checklist

`scripts/new_op.sh <name>` scaffolds the boilerplate; you fill in the math:

1. Declare the signature in `include/blt/ops/<name>.h`.
2. Implement `<name>_cpu.c` in `src/backend_cpu/`.
3. Write `tests/unit/backend_cpu/<name>_cpu_test.c` against a Python-dumped reference.
4. Wire it into `src/core/backend.c`'s dispatch.
5. _(Optional, anytime)_ Implement `<name>_cuda.cu`, add `tests/parity/<name>_parity_test.c`.
6. _(Later)_ Add `bench/micro/<name>_bench.c` once you need its throughput number.

Steps 1–4 never require a GPU — the CUDA step is always optional and always last.

---

## 6. Ablation configs — how Phase 5 actually works mechanically

The BLT paper's ablation section (§7) is effectively a list of config knobs, not new modules:

|Knob|Where it lives|Paper reference|
|---|---|---|
|Hash n-gram sizes (e.g. `3,4,5` vs `6,7,8` vs both)|`hash_ngram` config block|Table 8|
|Per-n-gram hash vocab size (100k/200k/400k)|same|Table 8|
|Cross-attention placement (encoder: none/last-layer/all-layers; pooling-init on/off; decoder: none/first-layer/all-layers)|`local_encoder`/`local_decoder` config block|Table 7|
|Entropy model size and context window|`entropy_model` config block|Figure 8|
|Local encoder/decoder layer split (e.g. 1 encoder / 9 decoder vs 5/5)|`local_encoder`/`local_decoder` config block|Table 9|
|Patch size / patching scheme (global-threshold vs monotonic, average patch size)|`patcher` config block|§2.3, Table 6|

Because none of these change function signatures — only which config values a fixed set of modules read — `tools/ablation_sweep.c` can run every variant in `configs/ablations/` back-to-back against the same held-out data, log bits-per-byte and the FLOPs/throughput numbers from `bench/ablation/`, and produce a single comparison table without touching model code between runs. `docs/ablations.md` is the running log of what you tried and what happened — treat it like a lab notebook, not a formal report.

---

## 7. How phases map onto the tree

|Phase|New directories/files that appear|Nothing changes in...|
|---|---|---|
|0|`core/`, `backend_cpu/` (basic ops), `tests/test_main.c`|—|
|1|`model/entropy_model.*`|`core/`, `backend_cpu/`|
|2|`model/patcher.*`, `data/byte_stream.*`, `tools/calibrate_threshold.c`|everything above|
|3|`model/hash_ngram.*`, `model/local_encoder.*`, `ops/cross_attention.h` gains real content|everything above|
|4|`model/patch_transformer.*`, `model/local_decoder.*`, `tools/flops_calculator.c`|everything above|
|5|`configs/ablations/*`, `tools/ablation_sweep.c`, `bench/ablation/`, `docs/ablations.md`|**no new source files in `model/`** — this phase is config sweeps + benchmark comparisons over Phase 3/4 code|
|6|`infer/self_speculation.*`, `infer/block_diffusion.*` (optional)|`model/` files unchanged — this phase only touches `infer/`|
|7|`backend_cuda/*` filled in progressively, `attention_varlen_cuda.cu` added|`include/` headers unchanged|
|8|`data/code_corpus.*`, `tools/code_eval.c`|everything above|

Phase 5 is deliberately "no new model files" — that's the point of having ablation knobs be config from the start. If Phase 5 ever requires editing `model/*.c` beyond exposing a new config field, that's a sign a knob should have been made configurable earlier.

---

## 8. Minimal `CMakeLists.txt` skeleton

```cmake
cmake_minimum_required(VERSION 3.20)
project(blt C)

option(BLT_WITH_CUDA "Build CUDA backend" OFF)
option(BLT_ASAN "Enable ASan/UBSan (debug builds)" ON)

add_library(blt_core src/core/tensor.c src/core/allocator.c src/core/backend.c)
add_library(blt_backend_cpu
    src/backend_cpu/matmul_cpu.c
    src/backend_cpu/rmsnorm_cpu.c
    # ... one line per file in backend_cpu/
)
target_link_libraries(blt_core PUBLIC blt_backend_cpu)

if(BLT_WITH_CUDA)
    enable_language(CUDA)
    add_library(blt_backend_cuda
        src/backend_cuda/matmul_cuda.cu
        src/backend_cuda/rmsnorm_cuda.cu
        # ...
    )
    target_link_libraries(blt_core PUBLIC blt_backend_cuda)
    target_compile_definitions(blt_core PUBLIC BLT_WITH_CUDA)
endif()

add_library(blt_model
    src/model/entropy_model.c
    src/model/patcher.c
    src/model/hash_ngram.c
    src/model/local_encoder.c
    src/model/patch_transformer.c
    src/model/local_decoder.c
)
target_link_libraries(blt_model PUBLIC blt_core)

add_executable(blt_tests tests/test_main.c)
target_link_libraries(blt_tests PRIVATE blt_model)

add_executable(blt_train tools/train.c)
target_link_libraries(blt_train PRIVATE blt_model)
```

Two build profiles you'll use constantly:

```bash
# CPU debug — Phases 0-5, everyday correctness and ablation work
cmake -B build-debug -DBLT_WITH_CUDA=OFF -DCMAKE_BUILD_TYPE=Debug -DBLT_ASAN=ON
cmake --build build-debug

# CUDA release — Phase 7 onward
cmake -B build-cuda -DBLT_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda
```

---

## 9. Naming conventions

- Public functions: `blt_<module>_<verb>` (`blt_rmsnorm_forward`, `blt_patcher_segment`).
- Every forward has a matching `_backward` declared from day one, even before it's implemented.
- Tensor shape convention, fixed once in `core/tensor.h`: `[batch, seq, dim]` for both byte- and patch-level tensors (there are only two tiers now, which simplifies this compared to a three-tier design).
- Config files (`configs/*.json`) are the only place hyperparameters live — including every ablation knob from §6. No magic numbers in `model/*.c`.