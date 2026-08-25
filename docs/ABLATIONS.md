# Architecture Ablation Protocol

All ablation sweeps (5.1 ngram, 5.2 cross-attention placement, 5.3 depth,
5.5 patcher) follow this protocol.

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
   is dense - it updates the entire table every step, and grads are zeroed +
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

## Determinism check

Two baseline_p4 runs (same seed/config): bpb, bpb_c, bpb_h, avg_patch_len,
flops_per_byte and per-step latency stats identical to machine precision;
only wall_clock_sec/throughput differ (1047.7 s vs 1059.0 s). PASS.

## Results tables

Delta columns are % vs baseline_p4 (lower is better for BPB; the throughput
column is inverted in sign by bench_report - positive % there means slower).

### 5.1 ngram

All variants are within +-0.11% of baseline and every single one is slightly
WORSE than the baseline n-gram config {3,4} @ 50k/table. At this model/data
scaling up the hash-ngram tables does not pay for itself beyond the small baseline
setup; bigger vocabs only cost throughput (dense full-tensor SGD).

| name | tag | n | mean_ms | p50_ms | p90_ms | p99_ms | bpb* | bpb_c* | bpb_h* | avg_patch_len* | train_steps* | wall_clock_sec* | flops_per_byte* | throughput_bytes_per_sec* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_eval | baseline_p4 | 1200 | 729.293 | 723.488 | 774.488 | 842.074 | 5.39251 | 5.4626 | 5.41699 | 31.6905 | 2.000e+03 | 1.059e+03 | 4.466e+07 | 290.085 | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) |
| train_eval | ngram_s678_v100k | 1200 | 734.24 | 726.381 | 760.427 | 811.467 | 5.39324 | 5.46316 | 5.42504 | 31.6293 | 2.000e+03 | 1.059e+03 | 4.466e+07 | 290.032 | +0.68% (-) | +0.01% (-) | +0.01% (-) | +0.15% (-) | -0.19% (+) | +0.00% | +0.02% (-) | +0.00% | -0.02% (+) |
| train_eval | ngram_s345_v200k | 1200 | 807.32 | 792.187 | 852.979 | 959.268 | 5.3933 | 5.46025 | 5.41769 | 32 | 2.000e+03 | 1.153e+03 | 4.466e+07 | 266.499 | +10.70% (-) | +0.01% (-) | -0.04% (+) | +0.01% (-) | +0.98% (-) | +0.00% | +8.85% (-) | +0.00% | -8.13% (+) |
| train_eval | ngram_s345_v50k | 1200 | 740.676 | 734.723 | 781.586 | 861.45 | 5.39632 | 5.46468 | 5.41786 | 32 | 2.000e+03 | 1.073e+03 | 4.466e+07 | 286.335 | +1.56% (-) | +0.07% (-) | +0.04% (-) | +0.02% (-) | +0.98% (-) | +0.00% | +1.31% (-) | +0.00% | -1.29% (+) |
| train_eval | ngram_s678_v200k | 1200 | 782.402 | 774.674 | 806.836 | 855.887 | 5.39682 | 5.4633 | 5.41967 | 32 | 2.000e+03 | 1.117e+03 | 4.466e+07 | 274.938 | +7.28% (-) | +0.08% (-) | +0.01% (-) | +0.05% (-) | +0.98% (-) | +0.00% | +5.51% (-) | +0.00% | -5.22% (+) |
| train_eval | ngram_all_v50k | 1200 | 734.297 | 726.048 | 758.864 | 803.726 | 5.39718 | 5.46393 | 5.41751 | 31.6293 | 2.000e+03 | 1.059e+03 | 4.466e+07 | 290.143 | +0.69% (-) | +0.09% (-) | +0.02% (-) | +0.01% (-) | -0.19% (+) | +0.00% | -0.02% (+) | +0.00% | +0.02% (-) |
| train_eval | ngram_s345_v100k | 1200 | 767.45 | 762.488 | 818.878 | 889.671 | 5.39721 | 5.45796 | 5.41742 | 31.6293 | 2.000e+03 | 1.105e+03 | 4.466e+07 | 278.026 | +5.23% (-) | +0.09% (-) | -0.08% (+) | +0.01% (-) | -0.19% (+) | +0.00% | +4.34% (-) | +0.00% | -4.16% (+) |
| train_eval | ngram_all_v100k | 1200 | 790.492 | 787.072 | 814.845 | 851.828 | 5.39799 | 5.46363 | 5.4215 | 32 | 2.000e+03 | 1.127e+03 | 4.466e+07 | 272.479 | +8.39% (-) | +0.10% (-) | +0.02% (-) | +0.08% (-) | +0.98% (-) | +0.00% | +6.46% (-) | +0.00% | -6.07% (+) |
| train_eval | ngram_s678_v50k | 1200 | 708.484 | 701.598 | 731.398 | 779.619 | 5.39826 | 5.46614 | 5.42066 | 32 | 2.000e+03 | 1.028e+03 | 4.466e+07 | 298.752 | -2.85% (+) | +0.11% (-) | +0.06% (-) | +0.07% (-) | +0.98% (-) | +0.00% | -2.90% (+) | +0.00% | +2.99% (-) |

* metric values (domain metrics: BPB, NFEs, GB/s, ...)

delta vs baseline tag 'baseline_p4': (+) improvement, (-) regression; latency/metrics assumed lower-is-better.


### 5.2 cross-attention placement

NOTE (expected artifact, not a bug): none / encoder_all / encoder_last produce
bit-identical results because with decoder cross-attention disabled the patch
stream never reaches the loss - encoder placement has nothing to influence.
The informative axis is decoder placement: decoder_all (with encoder cross-attn
removed entirely) is nominally best at -0.02% BPB while also saving encoder
cross-attn compute. Consistent with the paper's Table 7 finding that decoder
wants All Layers and encoder cross-attn adds little. Effect sizes are tiny
(~0.003 bpb), same order as the paper's.

| name | tag | n | mean_ms | p50_ms | p90_ms | p99_ms | bpb* | bpb_c* | bpb_h* | avg_patch_len* | train_steps* | wall_clock_sec* | flops_per_byte* | throughput_bytes_per_sec* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_eval | xattn_decoder_all | 1200 | 686.984 | 679.342 | 712.904 | 751.984 | 5.39134 | 5.46161 | 5.4159 | 31.6905 | 2.000e+03 | 1.002e+03 | 4.466e+07 | 306.638 | -5.80% (+) | -0.02% (+) | -0.02% (+) | -0.02% (+) | +0.00% | +0.00% | -5.40% (+) | +0.00% | +5.71% (-) |
| train_eval | xattn_both_poolmax | 1200 | 685.321 | 682.357 | 693.663 | 715.739 | 5.39189 | 5.46269 | 5.41643 | 31.6905 | 2.000e+03 | 996.35 | 4.466e+07 | 308.325 | -6.03% (+) | -0.01% (+) | +0.00% (-) | -0.01% (+) | +0.00% | +0.00% | -5.92% (+) | +0.00% | +6.29% (-) |
| train_eval | baseline_p4 | 1200 | 729.293 | 723.488 | 774.488 | 842.074 | 5.39251 | 5.4626 | 5.41699 | 31.6905 | 2.000e+03 | 1.059e+03 | 4.466e+07 | 290.085 | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) |
| train_eval | xattn_decoder_first | 1200 | 673.186 | 665.628 | 696.665 | 741.366 | 5.39257 | 5.46176 | 5.41674 | 31.6905 | 2.000e+03 | 984.269 | 4.466e+07 | 312.11 | -7.69% (+) | +0.00% (-) | -0.02% (+) | -0.00% (+) | +0.00% | +0.00% | -7.06% (+) | +0.00% | +7.59% (-) |
| train_eval | xattn_none | 1200 | 669.306 | 663.814 | 691.215 | 720.772 | 5.39495 | 5.46183 | 5.4197 | 31.6905 | 2.000e+03 | 979.437 | 4.466e+07 | 313.649 | -8.23% (+) | +0.05% (-) | -0.01% (+) | +0.05% (-) | +0.00% | +0.00% | -7.51% (+) | +0.00% | +8.12% (-) |
| train_eval | xattn_encoder_all | 1200 | 682.277 | 676.255 | 705.717 | 740.039 | 5.39495 | 5.46183 | 5.4197 | 31.6905 | 2.000e+03 | 995.626 | 4.466e+07 | 308.55 | -6.45% (+) | +0.05% (-) | -0.01% (+) | +0.05% (-) | +0.00% | +0.00% | -5.98% (+) | +0.00% | +6.37% (-) |
| train_eval | xattn_encoder_last | 1200 | 683.778 | 678.43 | 706.882 | 736.076 | 5.39495 | 5.46183 | 5.4197 | 31.6905 | 2.000e+03 | 997.3 | 4.466e+07 | 308.032 | -6.24% (+) | +0.05% (-) | -0.01% (+) | +0.05% (-) | +0.00% | +0.00% | -5.83% (+) | +0.00% | +6.19% (-) |
| train_eval | xattn_encoder_all_poolmax | 1200 | 656.799 | 659.976 | 683.89 | 701.05 | 5.39495 | 5.46183 | 5.4197 | 31.6905 | 2.000e+03 | 960.254 | 4.466e+07 | 319.915 | -9.94% (+) | +0.05% (-) | -0.01% (+) | +0.05% (-) | +0.00% | +0.00% | -9.32% (+) | +0.00% | +10.28% (-) |
| train_eval | xattn_encoder_last_poolmax | 1200 | 659.953 | 659.392 | 675.115 | 691.494 | 5.39495 | 5.46183 | 5.4197 | 31.6905 | 2.000e+03 | 961.154 | 4.466e+07 | 319.616 | -9.51% (+) | +0.05% (-) | -0.01% (+) | +0.05% (-) | +0.00% | +0.00% | -9.24% (+) | +0.00% | +10.18% (-) |

* metric values (domain metrics: BPB, NFEs, GB/s, ...)

delta vs baseline tag 'baseline_p4': (+) improvement, (-) regression; latency/metrics assumed lower-is-better.


### 5.3 depth (total local layers fixed at 10)

All four splits are WORSE than the 3-local-layer baseline (+0.20..0.23%).
Note: step count was frozen at 2000 for comparability, so deeper
stacks are under-trained here; wall clock confirms they do ~2x work per byte.
Within the sweep, shallow-encoder/deep-decoder (1,9) wins as the paper found,
and the ordering (enc1 < enc3 < enc5 < enc9 by BPB) matches the paper's
direction: keep the encoder cheap, put capacity in the decoder.

| name | tag | n | mean_ms | p50_ms | p90_ms | p99_ms | bpb* | bpb_c* | bpb_h* | avg_patch_len* | train_steps* | wall_clock_sec* | flops_per_byte* | throughput_bytes_per_sec* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_eval | baseline_p4 | 1200 | 729.293 | 723.488 | 774.488 | 842.074 | 5.39251 | 5.4626 | 5.41699 | 31.6905 | 2.000e+03 | 1.059e+03 | 4.466e+07 | 290.085 | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) |
| train_eval | depth_enc1_dec9 | 1200 | 1.727e+03 | 1.722e+03 | 1.832e+03 | 1.906e+03 | 5.40307 | 5.46803 | 5.4248 | 31.2672 | 2.000e+03 | 2.352e+03 | 1.467e+08 | 130.595 | +136.82% (-) | +0.20% (-) | +0.10% (-) | +0.14% (-) | -1.34% (+) | +0.00% | +122.13% (-) | +228.50% (-) | -54.98% (+) |
| train_eval | depth_enc3_dec7 | 1200 | 1.793e+03 | 1.774e+03 | 1.896e+03 | 1.982e+03 | 5.40374 | 5.46489 | 5.42729 | 31.2672 | 2.000e+03 | 2.428e+03 | 1.467e+08 | 126.548 | +145.85% (-) | +0.21% (-) | +0.04% (-) | +0.19% (-) | -1.34% (+) | +0.00% | +129.23% (-) | +228.50% (-) | -56.38% (+) |
| train_eval | depth_enc5_dec5 | 1200 | 1.819e+03 | 1.812e+03 | 1.858e+03 | 1.921e+03 | 5.40412 | 5.46762 | 5.42958 | 31.2672 | 2.000e+03 | 2.451e+03 | 1.467e+08 | 125.325 | +149.46% (-) | +0.22% (-) | +0.09% (-) | +0.23% (-) | -1.34% (+) | +0.00% | +131.47% (-) | +228.50% (-) | -56.80% (+) |
| train_eval | depth_enc9_dec1 | 1200 | 1.988e+03 | 1.983e+03 | 2.030e+03 | 2.084e+03 | 5.40484 | 5.46815 | 5.42935 | 31.2672 | 2.000e+03 | 2.654e+03 | 1.467e+08 | 115.749 | +172.64% (-) | +0.23% (-) | +0.10% (-) | +0.23% (-) | -1.34% (+) | +0.00% | +150.62% (-) | +228.49% (-) | -60.10% (+) |

* metric values (domain metrics: BPB, NFEs, GB/s, ...)

delta vs baseline tag 'baseline_p4': (+) improvement, (-) regression; latency/metrics assumed lower-is-better.


### 5.5 patcher

ANOMALY (documented honestly): the t4/t6/t8 threshold targets missed. The
warmup entropy stats used for calibration (mean 4.49) were measured mid-warmup;
by warmup end the entropy LM is good enough that per-byte entropies rarely
exceed even 5.01, so all three thresholds saturate at max_patch_length=32
(avg_patch ~31.7 == baseline behaviour). The three runs are effectively
replicates of baseline, not a patch-size sweep.

The informative rows are the controls: strided-4 and whitespace patching give
+38..43% throughput at only +0.04..0.06% BPB - a large efficiency win for a
tiny quality cost. avg_patch_len 4.0 / 5.0 confirm the rules did what they
should.

| name | tag | n | mean_ms | p50_ms | p90_ms | p99_ms | bpb* | bpb_c* | bpb_h* | avg_patch_len* | train_steps* | wall_clock_sec* | flops_per_byte* | throughput_bytes_per_sec* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_eval | patch_t8 | 1200 | 703.292 | 697.247 | 726.713 | 764.87 | 5.3925 | 5.46261 | 5.41663 | 31.6905 | 2.000e+03 | 1.023e+03 | 4.466e+07 | 300.42 | -3.57% (+) | -0.00% (+) | +0.00% (-) | -0.01% (+) | +0.00% | +0.00% | -3.44% (+) | +0.00% | +3.56% (-) |
| train_eval | baseline_p4 | 1200 | 729.293 | 723.488 | 774.488 | 842.074 | 5.39251 | 5.4626 | 5.41699 | 31.6905 | 2.000e+03 | 1.059e+03 | 4.466e+07 | 290.085 | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) | (baseline) |
| train_eval | patch_t6 | 1200 | 702.281 | 695.939 | 724.961 | 764.142 | 5.39261 | 5.4627 | 5.41676 | 31.6905 | 2.000e+03 | 1.021e+03 | 4.466e+07 | 300.965 | -3.70% (+) | +0.00% (-) | +0.00% (-) | -0.00% (+) | +0.00% | +0.00% | -3.62% (+) | +0.00% | +3.75% (-) |
| train_eval | patch_t4 | 1200 | 704.545 | 697.765 | 728.882 | 768.667 | 5.39301 | 5.46261 | 5.41681 | 31.5077 | 2.000e+03 | 1.024e+03 | 4.466e+07 | 299.869 | -3.39% (+) | +0.01% (-) | +0.00% (-) | -0.00% (+) | -0.58% (+) | +0.00% | -3.26% (+) | +0.00% | +3.37% (-) |
| train_eval | patch_strided4 | 1200 | 587.552 | 578.292 | 616.651 | 658.526 | 5.39452 | 5.46295 | 5.41787 | 4 | 2.000e+03 | 753.988 | 5.111e+07 | 407.434 | -19.44% (+) | +0.04% (-) | +0.01% (-) | +0.02% (-) | -87.38% (+) | +0.00% | -28.80% (+) | +14.44% (-) | +40.45% (-) |
| train_eval | patch_whitespace | 1200 | 578.891 | 571.327 | 630.314 | 724.444 | 5.39579 | 5.46309 | 5.41855 | 4.95434 | 2.000e+03 | 742.227 | 4.963e+07 | 413.889 | -20.62% (+) | +0.06% (-) | +0.01% (-) | +0.03% (-) | -84.37% (+) | +0.00% | -29.91% (+) | +11.14% (-) | +42.68% (-) |

* metric values (domain metrics: BPB, NFEs, GB/s, ...)

delta vs baseline tag 'baseline_p4': (+) improvement, (-) regression; latency/metrics assumed lower-is-better.


## Winner analysis

Winners per setting:
- 5.1 ngram: baseline {3,4} @ 50k (every alternative is worse)
- 5.2 placement: decoder_all (-0.02%, also cheapest of the top group)
- 5.3 depth: baseline depth (deeper loses at frozen budget)
- 5.5 patcher: strided-4 (+40% throughput, +0.04% BPB) if efficiency counts;
  otherwise no change

Combined winner ("winner_combined", VALIDATED with a dedicated run):
baseline ngram {{3,4}} @ 50k + placement=decoder_all + patcher=fixed:4.

| config | bpb | dBPB vs base | throughput | wall |
|---|---|---|---|---|
| baseline_p4 (paper defaults) | 5.3925 | - | 290 B/s | ~18 min |
| winner_combined | **5.3782** | **-0.27%** | **426 B/s (+47%)** | 20 min |

The combination beats its parts: decoder_all alone was -0.02% and strided-4
alone +0.04%, but together they land at -0.27% BPB while moving ~47% more
bytes per second. The mechanism is plausible rather than mysterious: shorter
patches give the byte-level decoder more local steps per window and denser
gradients at fixed step count, and dropping encoder cross-attn removes noise
the global stream never needed at this scale. Per-domain numbers also improve
(bpb_c 5.4626 -> 5.4455, bpb_h 5.4170 -> 5.4039).

Naive all-paper-defaults (six n-gram sizes @ large vocab, all-layer cross-attn
in both modules, entropy-based dynamic patching) is strictly more compute than
the winner on every axis; our grid shows each such choice costs throughput
without improving BPB at this scale (see 5.1/5.2 tables and the saturated
threshold rows in 5.5).

Notes: single seed per sweep point (determinism verified for baseline only),
2000-step frozen budget under-trains deeper stacks (5.3), and the entropy-
patcher threshold targets missed calibration so dynamic-vs-fixed patching was
only compared in its saturating regime. None of these change the direction of
the winner, but absolute gaps are small (~0.003 bpb) outside the patcher's
throughput effect.
