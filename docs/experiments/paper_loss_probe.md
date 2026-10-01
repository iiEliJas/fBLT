# Paper-Exact Loss: 30k Probe and Layout Re-measurement

Follow-up to `loss_normalization.md`. A 30k-step run under the paper-exact loss
(`--paper-loss 1`, AdamW `lr 1e-4`, `mask_scale 1.0`, `max_norm 20000`) on
TinyStories (2 enc / 6 global / 2 dec, `W=512`, `B=4`, entropy patches), followed
by a `masked_acc` evaluation of the resulting checkpoint.

## Headline

At 30k steps the model reaches `causal_bpb 1.7536` and train-layout masked
accuracy `0.518`, versus `5.3614` / `0.089` for the old mean-form SGD run at a
comparable budget. That is a **5.8x improvement in masked accuracy** and a
**3.1x improvement in BPB**.

This is the paper-exact loss working as intended. It was not a marginal gain: the
loss objective was materially wrong before, not just non-canonical.

## Training was stable, and the clip went inert

Loss fell `2041 -> 1142` over 30k with modest fluctuation (the per-step `t`
resampling), and gradient norms did not grow across thirds (`3814 -> 4481`).

Critically, **0 of 30 sampled steps exceeded `max_norm 20000`** (max gradient
`9318`). Under the old mean-form loss, 100% of steps clipped by a factor
swinging `2.6x` to `44772x`. The paper form makes the clip effectively inert, so
Adam's own normalization is doing the work and no variable rescale is corrupting
the moment estimates. This supports the hypothesis in `loss_normalization.md`
that the earlier loss instability was largely the clipping interaction.

## Layout re-measurement (and two corrected conclusions)

The 30k checkpoint was evaluated with `masked_acc --layout novel` over 32 windows,
2 sites each. Two findings **reverse** what the undertrained 20k model showed:

### 1. Aligned vs midpatch gap is now LARGE (Step 6 is triggered)

| t | aligned | midpatch | delta | 95% CI |
|---|---|---|---|---|
| 1.00 | 0.3164 | 0.1445 | **-0.1719** | [-0.2433, -0.1005] |
| 0.50 | 0.4408 | 0.1974 | **-0.2434** | [-0.3446, -0.1423] |

Both intervals exclude zero. The earlier 20k measurement had CIs that *included*
zero, so I concluded the lagging latent was free and Step 6 was not needed. That
conclusion was an artifact of an undertrained model, which could not exploit the
lagged latent either way. At 30k the lagging latent costs **17 points** at
t=1.0.

**Implication:** inference should start diffusion blocks only where
`blt_next_starts_patch` says a patch starts, finishing the open patch with
ordinary AR steps first. The earlier "no large gap, skip Step 6" advice is
superseded.

### 2. The `o_{i-1}` convention matters far more than previously measured

Cross-attention target `o_{i-1}` (paper) vs the last closed latent:

| t | delta | 95% CI |
|---|---|---|
| 1.00 | **-0.0625** | [-0.0731, -0.0518] |
| 0.50 | **-0.0955** | [-0.1128, -0.0782] |

The undertrained model measured `-0.0056`; the trained model measures `-0.06` to
`-0.095`. Using the last closed latent instead of `o_{i-1}` costs 6-10 accuracy
points. The paper's convention is not a wash, as the weak-model numbers
suggested.

### 3. No leak, and the novel layout is not degraded

Train `0.2686` vs aligned-novel `0.3164` at t=1.0 (gap `-0.0478`), consistent with
the leak fix holding. The aligned novel layout now slightly *exceeds* the trained
layout, which the weak model never did.

## Caveats

- 32 windows / 2 sites for the layout table: the `aligned`/`midpatch`/`train_lastlat`
  cells have `n = 152-256` per `t`, so their CIs are wide (+-0.07). The train
  rows (`n > 6000`) are precise. The midpatch and convention gaps are large
  enough to survive this; a full 208-window run would tighten them.
- The 600k run has not been done yet; all quality numbers here are from 30k steps
  and will improve. The *relative* findings (leak closed, `o_{i-1}` wins,
  midpatch costly) should hold, but the absolute accuracies will rise.