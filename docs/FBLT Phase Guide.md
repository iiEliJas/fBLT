
A detailed, build-order walkthrough for a plain BLT + Fast-BLT model, targeting code completion / small general LLM use. Each phase lists **goal**, **why it's built this way**, **what to implement**, **files touched** (paths from `BLT_PROJECT_STRUCTURE.md`), **tests**, **benchmarks**, and **exit criteria**.

Treat exit criteria as hard gates — the most common way this kind of project stalls is debugging two phases' worth of problems at once.

---

## Phase 0 — Foundations

**Goal:** a tensor library and testing harness you trust completely, before any ML code exists.

**Why this order:** every later bug-hunt assumes the tensor/allocator layer is _not_ the source of the bug. Earn that assumption once, here.

**Build:**

- `blt_tensor`: raw buffer + shape + strides + dtype + backend tag, flat C struct.
- Arena/bump allocator (`core/allocator.c`): one arena per training step, reset (not freed) at step boundaries.
- `core/backend.c` dispatch skeleton — wire it up now even with only one backend, so the pattern is established before there's a second to keep in sync.
- Basic ops in `backend_cpu/`: elementwise add/mul, matmul (naive triple loop is fine), softmax.
- `tests/test_main.c`: your own harness, a registry of `(name, fn)` pairs plus a runner reporting pass/fail/timing. Keep it under ~150 lines.
- `tools/dump_reference_io.py`: given an op name, generates random weights + inputs, computes the NumPy/PyTorch reference output, dumps all three to a flat binary. This is the backbone of every correctness test for the rest of the project.

**Files:** `include/blt/core/*`, `src/core/*`, `src/backend_cpu/{matmul,elementwise,softmax}_cpu.c`, `tests/test_main.c`, `tools/dump_reference_io.py`.

**Tests:** matmul/add/softmax vs. Python reference, `|Δ| < 1e-4` fp32; allocator stress test, ASan clean.

**Benchmarks:** none yet — confirm the harness runs and reports timing correctly.

**Exit criteria:** every Phase 0 op passes its reference-diff test; ASan/UBSan build is clean; you can add a brand-new op end-to-end in under 15 minutes.

---

## Phase 1 — Byte-level entropy model

**Goal:** a tiny causal byte LM whose next-byte entropy will drive patch boundaries (BLT §2.3, §4.2).

**Why this order:** small, self-contained, no dependency on patching or encoders — the cheapest place to shake out bugs in RoPE/attention/norm/FFN before those primitives get reused everywhere else.

**Build:**

- Byte embedding (256-ish vocab).
- RoPE (`ops/rope.h`), RMSNorm (`ops/rmsnorm.h`), SwiGLU FFN.
- Causal self-attention with a sliding window (`ops/attention.h`) — build this generically now; the local encoder, patch transformer, and decoder all reuse the same masked-MHA op with different masks.
- Cross-entropy loss + backward.
- Start small: 2–4 layers, dim 128, window 512. BLT's eventual 100M/14-layer entropy model is a config change later, not new code.

**Files:** `include/blt/ops/{rope,rmsnorm,attention}.h`, `src/backend_cpu/{rope,rmsnorm,attention}_cpu.c`, `src/model/entropy_model.c`.

**Tests:**

- RoPE/RMSNorm/SwiGLU/masked-attention forward+backward vs. Python reference.
- Finite-difference gradient check on the whole tiny model — once per op class, catches most backward-pass bugs.
- Overfit-one-batch integration test: 8 repeated short byte strings, loss → ~0 within a few hundred steps.

**Benchmarks:** none required. If you want early CUDA practice, matmul/softmax/RMSNorm kernels are safe to write in parallel here, diffed against Phase 0's CPU references.

**Exit criteria:** overfit test passes; gradient check passes for every op class; visibly decreasing bits-per-byte on a small real corpus slice (a code-heavy slice is a good idea now, since it's your eventual target).

---

## Phase 2 — Dynamic entropy patcher

**Goal:** implement BLT's boundary rules — global threshold and approximate-monotonic constraint (§2.3, Eq. 1).

**Why this order:** pure, stateless preprocessing with no learned weights of its own. Bugs here are cheap to isolate (deterministic input → deterministic output) and _must_ be caught before Phase 3, since a wrong boundary silently corrupts every downstream tier.

**Build:**

- Global threshold rule: boundary wherever $H(x_i) > \theta_g$.
- Approximate-monotonic rule: boundary wherever $H(x_i) - H(x_{i-1}) > \theta_r$.
- Max-patch-length cap.
- Entropy-context reset at newlines (BLT §4.4) — this matters _especially_ for code, which is full of exactly the repetitive structured patterns (indentation, boilerplate, repeated tokens) the paper found triggered entropy drift.
- `tools/calibrate_threshold.c`: offline pass finding $\theta_g$ for a target average patch size on a data sample (BLT §4.3).

**Files:** `include/blt/model/patcher.h`, `src/model/patcher.c`, `include/blt/data/byte_stream.h`.

**Tests:**

- **Golden test:** reproduce BLT's own worked example ("Daenerys Targaryen is in Game of Thrones...", Figure 4) with their exact entropy table, assert boundaries match exactly.
- **Property/fuzz tests** on random byte sequences: every byte in exactly one patch; no patch exceeds max length or is empty; deterministic given the same input.

**Benchmarks:** bytes/sec throughput of the patcher alone (`bench/micro/`) — a linear scan, so slowness here is an algorithmic bug, not a "needs CUDA" situation.

**Exit criteria:** golden test matches the paper exactly; property tests pass under fuzzing; calibration converges to a target patch size on real code data.

---

## Phase 3 — Local Encoder (bytes → patch embeddings)

**Goal:** turn a patched byte stream into one embedding per patch, using hash n-gram embeddings and cross-attention pooling (BLT §3.2).

**Why this order:** introduces the pooling primitive (cross-attention where queries come from a coarser grouping than keys/values) — get its masking exactly right here, since it's the piece with the most subtle failure mode in the whole architecture.

**Build:**

- Rolling polynomial hash (BLT Appendix C) over byte n-grams, $n = 3..8$; hash n-gram embeddings summed into byte embeddings (Eq. 2–4).
- Local encoder transformer, 1–3 layers, local block-causal window (can cross patch boundaries, cannot cross document boundaries).
- Patch cross-attention module (Eq. 5–8, Figure 5): query per patch initialized by pooling the patch's byte reps; keys/values are byte hidden states; mask restricts each query to its own patch's bytes only.
- **Expose every one of these as a config field now** (n-gram sizes, per-n-gram vocab size, encoder layer count, cross-attention pooling-init on/off) — this is what makes Phase 5's ablation sweep possible without touching this file again.

**Files:** `include/blt/model/hash_ngram.h`, `src/model/hash_ngram.c`, `include/blt/ops/cross_attention.h`, `src/backend_cpu/cross_attention_cpu.c`, `src/model/local_encoder.c`.

**Tests:**

- Hash determinism and rough bucket-distribution sanity check on a large random n-gram sample.
- **Mask correctness test:** synthetic patch boundaries, dump the cross-attention weight matrix, assert it's exactly block-structured (zero weight outside each patch's byte range). Highest-value test in this phase — a subtle off-by-one here corrupts every patch representation silently.
- Reference-check the full local encoder against a PyTorch reimplementation.

**Benchmarks:** none new beyond micro-level attention/matmul numbers if you've already started CUDA experiments.

**Exit criteria:** mask block-structure test passes exactly; reference-diff passes; hash embeddings show no pathological collisions on real code text.

---

## Phase 4 — Baseline BLT (global patch transformer + local decoder)

**Goal:** a complete, small, working byte-level LM. Your first real, trusted baseline.

**Why this order:** the last phase before you start iterating on architecture choices — you want a fully working reference to measure every later change against.

**Build:**

- Global patch transformer: standard block-causal decoder-only transformer over patches (BLT §3.1).
- Local decoder: cross-attention with queries/keys-values swapped relative to the encoder — byte reps are queries, patch reps are keys/values (Eq. 10–12, Figure 5's role-swap table). Implement as a distinct mode of the same `cross_attention` op, not a new op.
- Next-byte cross-entropy loss, end-to-end.
- **Expose decoder layer count, cross-attention placement (which layers, first/all), and global transformer size as config fields** — same reasoning as Phase 3.
- `tools/flops_calculator.c`: BLT's own FLOPs equations (Appendix B), for this exact architecture. Write it now while the model is small enough to hand-verify.

**Files:** `src/model/patch_transformer.c`, `src/model/local_decoder.c`, `tools/flops_calculator.c`.

**Tests:**

- Decoder cross-attention masking checked against BLT's Figure 5 role-swap spec.
- Overfit-one-batch on a handful of short documents/code snippets.
- End-to-end reference-diff (bytes in, byte-logits out) against a PyTorch reimplementation.

**Benchmarks:**

- Bits-per-byte on held-out data, trending down — your first real training curve, and the number every ablation in Phase 5 gets compared against.
- FLOPs-calculator sanity check against a hand-derived estimate at this scale.
- **Baseline profile:** where does wall-clock time actually go right now (entropy model / patcher / encoder / global transformer / decoder)? Record this — it's what tells you which ablation knobs in Phase 5 are worth trying first.

**Exit criteria:** overfit test passes; BPB beats a unigram-byte baseline on held-out data; FLOPs calculator matches a hand-derived estimate; you can greedily generate a short, non-garbage completion from a code prompt.

---

## Phase 5 — Architecture improvements from the paper's own ablations

**Goal:** use the BLT paper's ablation section (§7) as a direct menu of improvements, guided by where Phase 4's benchmarks say your model is actually weak or slow — rather than guessing.

**Why this phase exists as its own step:** the BLT authors already ran exactly this kind of sweep at their scale and published the results (Tables 6–9, Figure 8). You don't need to rediscover which knobs matter; you need to check whether their findings transfer to your scale and data (code, not general web/MT text), and pick up the free wins. This phase adds **no new modules** — see the project structure doc §6/§7 — it's config sweeps over Phase 3/4 code, run through `tools/ablation_sweep.c`.

**What to try, and what the paper found, in priority order (highest expected value first):**

1. **Hash n-gram embeddings** (Table 8) — the paper's single largest lever. Findings: smaller n-grams (3,4,5) matter more than larger ones (6,7,8); per-n-gram vocab size is the most significant parameter, with diminishing returns past ~300–400k hashes. For code, short n-grams over identifier characters and operators are plausibly even more valuable than in prose — worth checking specifically.
2. **Cross-attention placement** (Table 7) — findings: decoder cross-attention matters most; encoder cross-attention helps only with pooling-initialized queries; cross-attention helps particularly on structured/repetitive domains (their own Github-code numbers in Table 7 are the most directly relevant row to your use case — cross-check against it).
3. **Local encoder/decoder depth split** (Table 9) — findings: with hash n-gram embeddings in place, an extremely light encoder (1 layer) paired with a heavier decoder (7–9 layers) works well and is cheaper than an even split. Directly actionable for your parameter budget.
4. **Entropy model size and context window** (Figure 8) — findings: both help, with diminishing returns beyond ~50M params and context 512. If Phase 4's profiling shows the entropy model is a disproportionate share of wall-clock, this tells you where the ceiling is before you're just spending compute for nothing.
5. **Patch size / patching scheme** (Table 6, §5.1) — entropy patching outperforms static/whitespace patching, and larger average patch sizes trade a small quality cost for real inference FLOP savings (inference FLOPs are inversely proportional to average patch size). For a completion tool where latency matters, this is a direct lever on user-perceived speed — worth an explicit sweep of average patch size vs. measured BPB and vs. Phase 4's profiling numbers.

**Process, concretely:**

- For each knob: create a `configs/ablations/<name>.json` variant, run `tools/ablation_sweep.c` against a fixed held-out slice of your code corpus, log BPB + FLOPs/throughput to `bench/ablation/`, write the result and your interpretation into `docs/ablations.md`.
- Change **one knob at a time** relative to the Phase 4 baseline config — this is what makes the comparison table interpretable instead of a confound.
- Prioritize knobs your Phase 4 profiling flagged as expensive: if the local decoder dominates wall-clock, the depth-split and cross-attention-placement ablations matter more to you _right now_ than the entropy-model-size one, regardless of the paper's own ranking (their ranking was measured on their data/scale, not yours).

**Files:** `configs/ablations/*.json`, `tools/ablation_sweep.c`, `bench/ablation/*`, `docs/ablations.md`. No new files under `src/model/`.

**Tests:** none new beyond re-running the existing Phase 3/4 test suite against each new config (a config change should never break a mask-correctness or overfit test — if it does, that's a real bug, not an expected ablation tradeoff).

**Benchmarks:** this phase's entire output _is_ a benchmark comparison table — BPB and throughput/FLOPs for baseline vs. each variant, on the same held-out data.

**Exit criteria:** you have a comparison table (in `docs/ablations.md`) covering at least the top 3 knobs above; you've picked a "winning" config combining the improvements that actually helped on _your_ data (not just adopted the paper's exact numbers uncritically); the winning config passes all Phase 3/4 tests.

---

## Phase 6 — Fast decoding (Fast-BLT)

**Goal:** close the inference-speed gap from the local decoder emitting one byte at a time — directly serves "feels fast while typing" for code completion.

**Why this order:** inference-loop-only (BLT-S) or requires retraining just the decoder (BLT-D), independent of the CUDA port — can happen before or after Phase 7 CUDA work, but doing it here means your CUDA benchmarks in Phase 7 already reflect the faster decoding path.

**Build, in order of cost:**

1. **BLT-S (self-speculation)** — no architecture change: the local decoder autoregressively drafts $k$ bytes past the normal patch boundary, then one full forward pass through encoder/global/decoder verifies the draft, rolling back to the first mismatch (Fast-BLT Algorithm 2). Guaranteed byte-identical to plain greedy decoding. This is the priority pickup: code completion has a lot of highly predictable runs (closing brackets, repeated indentation, common idioms) where the decoder can safely draft far ahead before a mismatch.
2. **BLT-D (block diffusion decoder), optional stretch** — requires retraining the local decoder with a combined next-byte + masked-block-diffusion objective (Fast-BLT §3.2, Eq. 5–7) and new attention masks (causal on the clean prefix, bidirectional within each corrupted block, Figure 5). Bigger potential speedup, real added complexity, and a genuine retraining cost — only take this on once BLT-S's own numbers say more speed is worth it.

**Files:** `include/blt/infer/self_speculation.h`, `src/infer/self_speculation.c`, `src/infer/block_diffusion.c` (optional), `include/blt/infer/kv_cache.h` (two tiers now: byte-window, patch — simpler than a three-tier concept design would have needed).

**Tests:**

- BLT-S output-equivalence: greedy decode with and without speculation produces byte-identical output on the same prompt.
- Rollback correctness: construct a case with a known mismatch position, confirm the verified sequence matches a plain autoregressive run up to and including that position.
- (BLT-D) Attention mask test against the Fast-BLT Figure 5 spec.

**Benchmarks:** reproduce Fast-BLT's own metric set:

- Average decoder NFEs per generated sequence.
- Average encoder/global-model NFEs per generated sequence.
- Estimated memory bandwidth: $\frac{b[N_{dec}P_{dec} + N_{enc}(P_{enc}+P_{glob})]}{10^9}$ GB (Eq. 8), $b=2$ for bf16.
- Compare with vs. without BLT-S (and BLT-D if attempted) on a fixed held-out set of code completion prompts — this table is what tells you whether it's actually faster in the way that matters for your use case (latency per completion, not just raw NFE count).

**Exit criteria:** BLT-S output-equivalence passes on every test prompt; measured memory bandwidth drops meaningfully at equal completion quality; (if attempted) BLT-D's speed/quality tradeoff is characterized, not just a speed number in isolation.

---

## Phase 7 — CUDA port and scaling toward 1B

**Goal:** move from CPU-only, small-scale, verified model to a CUDA-accelerated model that reaches ~1B parameters.

**Why this order:** every design decision above was verified against a CPU reference specifically so this phase is a _port_, not a _rewrite_. Do this only once Phases 4–6 are numerically solid — debugging model correctness and kernel correctness simultaneously is exactly what the earlier CPU-first approach was meant to avoid.

**Build, per-op, following the checklist in the structure doc §5:**

- cuBLAS/cuBLASLt for GEMMs; custom kernels for RMSNorm, RoPE, softmax, cross-attention.
- Every ported kernel gets a `tests/parity/` diff against the already-verified CPU output (reusing the exact golden data from Phase 0/1, nothing new to generate).
- **Attention masking is the hard part**: patch boundaries differ per example. Start with pad + additive-mask (simple, correct, slower); only build a fused variable-length kernel (`attention_varlen_cuda.cu`) once profiling says this specific op is the bottleneck. BLT's own use of FlexAttention for this masking problem is worth reading as a reference for the mask patterns before designing your own kernel.
- Mixed precision: bf16 matmuls with fp32 accumulation.
- Two-tier KV cache (byte-window, patch) — needed for both plain generation and the BLT-S loop from Phase 6.
- Memory budget check before scaling to 1B: params × 4 bytes (fp32) + AdamW optimizer state; plan for bf16 weights + fp32 master weights ahead of time rather than discovering an OOM at the 1B run.

**Files:** `src/backend_cuda/*` filled in progressively, `include/blt/infer/kv_cache.h`, `src/infer/kv_cache.c`.

**Tests:** `tests/parity/` for every ported op, no exceptions; `compute-sanitizer` clean on a full training step; full end-to-end CPU-vs-CUDA diff at the model level (not just per-op) to catch integration-level mismatches.

**Benchmarks:**

- Micro: per-kernel achieved TFLOPs vs. cuBLAS/theoretical peak.
- Meso: per-tier forward+backward ms/throughput.
- Macro: full training step tokens/sec and memory; full generation loop NFEs/bandwidth (continuing Phase 6's metrics, now at real scale).
- **MFU tracking:** Phase 4's FLOPs calculator estimate vs. measured wall-clock, recorded at every scale-up step.
- Reach for Nsight Systems/Compute only once your own lightweight profiler has already pointed at a specific slow region.

**Exit criteria:** every op has a passing parity test; `compute-sanitizer` clean; a training step at target 1B config runs without OOM and without accuracy divergence vs. a matched small-scale CPU run; MFU is tracked and not mysteriously low.

---

## Phase 8 — Data pipeline and evaluation for code completion

**Goal:** infrastructure to train on real code corpora and to measure, concretely, whether the model is any good at completion — not just "the loss went down."

**Why this order:** listed last but built incrementally alongside Phases 4–7 in practice — you need real code data to meaningfully train past the tiny debug stage, and you want evaluation running continuously, not assembled retroactively.

**Build:**

- Precompute-and-cache pipeline (`tools/preprocess_corpus.c`): entropy trace → patch boundaries, written once to a packed binary shard format, not recomputed every epoch.
- `data/code_corpus.c`: repo-level dedup, language filtering, license filtering — code corpora have real data-hygiene problems (near-duplicate files, license contamination, generated/vendored code) that general text pipelines don't emphasize as much.
- Evaluation (`tools/code_eval.c`): bits-per-byte on held-out code; exact-match and edit-distance on held-out completions given a prefix; consider fill-in-the-middle (prefix+suffix→infix) style evaluation if you want completion quality closer to how real editors use these models — note this is a standard practice in code-model literature but not something either source paper covers, so treat it as your own addition, not a paper-derived requirement.
- Qualitative sampling: read actual completions regularly, from early in training. Numbers are unreliable before the pipeline is debugged; reading the completions usually isn't.

**Files:** `tools/preprocess_corpus.c`, `src/data/{shard_writer,shard_reader,code_corpus}.c`, `tools/code_eval.c`, `docs/metrics.md`.

**Tests:** shard round-trip (write then read reproduces exact byte/entropy/boundary data); `code_eval` metrics unit-tested against hand-computed small examples.

**Benchmarks:** shard read throughput (should comfortably exceed training step time); preprocessing throughput (bytes/sec) on a representative code corpus sample.

**Exit criteria:** a full epoch's worth of real code data preprocesses and caches without manual intervention; BPB, exact-match, and edit-distance are visible as live curves during training; qualitative completions on a fixed prompt set are checked by eye at regular intervals, not only at the very end.

---

## Suggested overall order, restated

Phases 0 → 1 → 2 → 3 → 4 (baseline BLT, CPU, small scale) → 5 (ablation-informed improvements, still CPU/small — cheap to iterate on before scaling) → 8-basics (enough data pipeline to train on real code) → 6.1 (BLT-S, cheap latency win with no architecture risk) → 7 (CUDA port, scale toward 1B) → 6.2 (BLT-D, optional) as a refinement once the full pipeline is trustworthy and benchmarks tell you it's worth the retraining cost.