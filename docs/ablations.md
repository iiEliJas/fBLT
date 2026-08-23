# Phase 5 Ablation Protocol (LOCKED)

All ablation sweeps (5.1 ngram, 5.2 cross-attention placement, 5.3 depth,
5.5 patcher) follow this frozen protocol. Any change to these rules invalidates
comparability and must be recorded here with a reason.

- Date locked: 2026-08-22
- Machine: Linux x86_64, 12 cores, 15 GiB RAM, CPU-only (no CUDA)
- Trainer: `./bin/train_sweep` (build with `make sweep`)
- Test suite at lock time: 44/44 green (`make test`)

## Data

Produced by `tools/prep_corpus.py` (seed=42) from
`data/raw/the-stack-smol/data/c/data.json` (10k files, bigcode/the-stack-smol).

- kept 9658 / 10000 files; rejected: 164 too_small, 171 too_large, 7 line_length,
  0 alphanum, 0 duplicate
- split 95/5 after deterministic shuffle

| stream | sha256 | bytes |
|---|---|---|
| train.bin | `ecacd854405c66ef5ee7a42d60c1991840a316a0dccbbcb74e797651cb37116a` | 67497634 |
| heldout.bin | `8dc522356f0ed2bf7690931c925c378b748e14f87eb01c75ef9a493a751d75e0` | 3686457 |
| heldout_c.bin | `275e66210a8ccaee76eded66efe12dfd0c967c3da0b8a27c6397ae990ed67173` | 1992799 |
| heldout_h.bin | `09e7d26ea907aa947cc651f34b2d8b9aba729cfb5a8c64c50ccb169ed4f27f14` | 1693658 |

## BPB definition (frozen)

BPB = sum over windows of shifted next-byte cross-entropy / (ln 2 * positions),
where positions = seq_len - 1 per window. The loss tensor from
`blt_model_forward` is the mean over shifted positions; the trainer re-weights
it by position count. Windows are carved per-file from the manifest index
(`floor(length / seq_len)` windows per file, tails dropped), so a window never
crosses a document boundary.

Eval subsets (fixed for all runs): first 64 windows of heldout.bin, and the
first 64 windows of each domain bin (`heldout_c.bin`, `heldout_h.bin`) for the
per-domain breakout. Eval runs every `eval_every=250` main steps and once at
the end; only the final numbers go to `bench/results.jsonl`.

## Training protocol (frozen)

- Windowing: sequential over the manifest-ordered train windows, wraparound;
  window index at step t is `t mod num_windows`.
- Entropy LM warmup: 800 steps, entropy-LM-only (own next-byte loss), same
  lr as main training. Runs before main step 0 in every config that uses an
  entropy patcher. Because seed + data order are identical everywhere, patch
  boundaries are identical across all sweep configs.
- Main training: exactly `steps=2000` steps total (800 warmup + 1200 main),
  constant lr=0.01 plain SGD, global-norm grad clipping at 1.0 (main model and
  entropy LM clipped independently).
- REVISION (2026-08-23): the first launch accidentally ran configs at
  steps=3000; those runs all tripped the 25-min wall guard and none ever
  wrote a results line, so the protocol was restored to steps=2000 with zero
  completed-run loss. All partial checkpoints/logs from the 3000-step era
  were discarded. The four depth_* configs run the same frozen 2000 steps but
  get max_wall_min=90 (guard is a safety net, not part of the protocol);
  they need ~40 min for identical work.
- seq_len=256, seed=42.
- Wall-clock guard: clean checkpoint + abort if a run exceeds 25 min
  (resume with `--resume bench/ckpts/<tag>.ckpt`).
- Determinism: xorshift64* RNG seeded from config; single-threaded math.
  Same config + seed must reproduce identical loss trajectories. Verified for
  baseline_p4 (two runs, trajectories compared from logs).
- Calibration: baseline_p4 measured ~140 s warmup + ~0.63 s/main step + evals
  ~= 16.5 min, inside the 15-18 min target; step count frozen there.

## Baseline (baseline_p4)

encoder: embed 96, 1 layer, ngram {3,4} @ 50k/table; global: embed 96, 2 layers;
decoder: embed 96, 2 layers; hidden_dim = 4*embed everywhere; heads = embed/32;
cross-attention placement "both" (all layers in encoder and decoder);
pooling_init MEAN; patcher rule global threshold 5.3 nats, max_patch_length 32.

## Documented deviations / caps

1. **Ngram vocab capped at 200k** (plan sketched up to 400k). `blt_sgd_step`
   is dense — it updates the entire table every step, and grads are zeroed +
   norm-scanned full-tensor too, so per-step cost is O(vocab * dim * tables)
   regardless of how few rows were touched. At 400k x dim 96 x 3 tables this
   adds >1 s/step and blows the frozen wall-clock budget. RAM would fit
   (~1 GB); time does not. Grid therefore uses {50k, 100k, 200k}.
2. **Entropy thresholds are absolute (nats)** on a co-trained entropy LM.
   The paper uses a pretrained entropy model; here it is warm-trained inside
   every run (800 steps, identical across configs) so comparisons stay fair.
   Thresholds for sweep 5.5 targets were derived from the post-warmup entropy
   distribution (mean ~4.49, sd ~0.77): t4 -> thr 5.01, t6 -> thr 5.24,
   t8 -> thr 5.37 (normal approximation, P(fire) = 1/target_len).
3. **5.4 entropy-model sweep skipped** pending profiling: the entropy LM costs
   ~0.175 s/warmup step vs ~0.63 s/main step (well under half of wall-clock),
   so per the plan's decision rule there is no meaningful wall-clock share to
   trade away. Revisit only if final profiles contradict this.

## Results tables

Filled in per sweep below (via `tools/bench_report.py --baseline baseline_p4`).

### 5.1 ngram

(pending)

### 5.2 cross-attention placement

(pending)

### 5.3 depth

(pending)

### 5.5 patcher

(pending)
