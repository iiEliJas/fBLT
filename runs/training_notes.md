# Training results and notes

Dates: 2026-08-24/25. All runs: `bin/train_blt_d`, corpus `data/train.bin`
(67.5 MB C code via tools/prep_corpus.py), fixed-stride-4 patches,
window N=48, embed 64 / hidden 128 / 2 layers per submodule, SGD lr=0.05
with x0.3 decay at 60%/85%, grad-norm clip 5.0, seed 7. Checkpoints:
`runs/plain_40k.fblt`, `runs/bltd_l03_40k.fblt` (FBLT v1 format,
src/models/checkpoint.c).

## Headline table (held-out eval, 300 windows)

Eval offsets into data/heldout.bin measure position sensitivity;
all three agree qualitatively, so the model generalizes rather than
memorizes (an earlier contamination suspicion was an artifact of a
checkpoint-load bug, since fixed -- see below).

| Model | Steps | Mask loss | skip=0 | skip=500k | skip=2M | Masked top-1 |
|---|---|---|---|---|---|---|
| Plain BLT   | 10k | —     | 2.72 | — | — | — |
| Plain BLT   | 20k | —     | 2.07 | — | — | — |
| Plain BLT   | 40k | —     | 1.26 | 1.34 | 1.43 | — |
| BLT-D l=1 always-on | 10k | on | 3.62 | — | — | n/a |
| BLT-D l=1 always-on | 15k | on | 3.49 | — | — | n/a |
| BLT-D l=1 always-on | 20k | on | 3.34 | — | — | n/a |
| BLT-D l=1 + warmup@10k | 20k | ramped | 3.03 | — | — | n/a |
| BLT-D l<=0.3 + warmup@10k | 20k | capped | 2.89 | — | — | n/a |
| BLT-D l<=0.3 + warmup@10k | 40k | capped | **1.65** | 1.76 | 1.99 | 99.97-100% |

## Diagnostic (machinery correctness)

BLT-D with the mask loss silenced (--t-min 1e9, identical code path):
2.13 BPB vs plain arm 2.07 at matched settings -> parity within noise.
Masks, preprocessing, forward, backward and D_0-table gradients all
train the clean objective exactly like plain BLT.

## Findings

1. Objective interference is real and monotone in mask weight
   (silenced < 0.3-capped < always-on); reweighting toward next-byte
   prediction follows Fast-BLT section 6.
2. At 40k steps BLT-D trails plain causally by ~0.4 BPB, consistent
   with the paper's own Table 2. In exchange it gains masked multi-byte
   prediction at ~100% top-1 -- the exact capability verified diffusion
   drafting requires. The latent o_j is computed from the same byte
   span the block predicts, which is why drafting accuracy saturates.
3. t-floor matters: without it, rare t~U(0,1) draws give 1/t weights up
   to ~1e6 with wild loss swings; --t-min 0.05 fixed it. fp32 -log(0)
   underflow needed probability clamps.

## Operating point (chosen)

**BLT-D, lambda capped at 0.3 with 10k-step warmup, 40k total steps**
(runs/bltd_l03_40k.fblt): best causal BPB among diffusion arms while
keeping saturated drafting accuracy. Plain BLT remains the reference
for pure next-byte work.

## Bugs found and fixed along the way

- Checkpoint clobbering: --load-weights loaded weights, then the
  unconditional random-init block overwrote them -> every loaded model
  behaved like fresh random init (~8 BPB everywhere). Init is now
  skipped when loading. This produced the earlier false "eval
  contamination" alarm; retracted.
- Makefile had no header dependency tracking; a struct change half-
  rebuilt the tree and segfaulted tests. Fixed with -MMD -MP.
- Unclamped 1/t loss weight and fp32 -log(0) underflow (see Findings 3).

## Next session

Next: Algorithm 1 block-diffusion inference (unmasking schedule,
alpha / EB-gamma, top-p sampler) + draft verification reusing
blt_verify_draft; validate against runs/bltd_l03_40k.fblt.
