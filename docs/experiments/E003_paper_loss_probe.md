# Paper-Exact Loss: 30k Probe and Layout Re-measurement

## Metadata

- **Experiment ID:** E003
- **Title:** Paper-exact loss 30k probe and layout re-measurement
- **Status:** Checked
- **Date or reporting period:** Not recorded in the historical report.
- **Author or contributors:** Not recorded in the historical report.
- **Related reports, issues, or papers:** [BLT-D loss normalization](E002_loss_normalization.md); Fast Byte Latent Transformer.
- **Git commit or release:** Not recorded in the historical report.
- **Configurations and scripts:** 30k-step run under `--paper-loss 1`, AdamW `lr 1e-4`, `mask_scale 1.0`, `max_norm 20000`, on TinyStories (2 enc / 6 global / 2 dec, `W=512`, `B=4`, entropy patches). Full command, seed, hardware, and evaluation commands are not recorded.
- **Artifacts:** Measurement tables are recorded in the report. Raw logs and checkpoints are not recorded.

## 1. Research question

Does the paper-exact loss improve masked accuracy at 30k steps, and do earlier layout
conclusions survive on a better-trained model?

## 2. Motivation and hypothesis

This follows [E002_loss_normalization.md](E002_loss_normalization.md). The hypothesis is that the
paper-exact loss is materially better than the mean-normalized form, not merely non-canonical.

## 3. Experimental design

- Trained a 30k-step model under the paper-exact loss.
- Evaluated causal BPB and train-layout masked accuracy.
- Evaluated layout variants with `masked_acc --layout novel` over 32 windows, 2 sites each.
- Compared against the older mean-form run at a comparable budget.

## 4. Reproduction instructions

Reproduction details are incomplete in this historical report. It records the configuration
and measurement tables, but not the full command, seed, hardware, or evaluation commands.

## 5. Results

### Headline

At 30k steps the model reaches causal BPB 1.7536 and train-layout masked accuracy 0.518,
versus 5.3614 / 0.089 for the old mean-form SGD run at a comparable budget. That is a 5.8x
improvement in masked accuracy and a 3.1x improvement in BPB.

### Training stability and clipping

Loss fell `2041 -> 1142` over 30k with modest fluctuation from per-step `t` resampling.
Gradient norms did not grow across thirds (`3814 -> 4481`).

Critically, 0 of 30 sampled steps exceeded `max_norm 20000` (max gradient 9318). Under the
old mean-form loss, 100% of steps clipped by a factor swinging 2.6x to 44772x. The paper
form makes the clip effectively inert, supporting the hypothesis that earlier instability was
largely the clipping interaction.

### Layout re-measurement

The 30k checkpoint was evaluated with `masked_acc --layout novel` over 32 windows, 2 sites
each. Two findings reverse what the undertrained 20k model showed.

#### 1. Aligned vs midpatch gap is now large

| t | aligned | midpatch | delta | 95% CI |
|---|---|---|---|---|
| 1.00 | 0.3164 | 0.1445 | -0.1719 | [-0.2433, -0.1005] |
| 0.50 | 0.4408 | 0.1974 | -0.2434 | [-0.3446, -0.1423] |

Both intervals exclude zero. The earlier 20k measurement had CIs that included zero, so the
conclusion that the lagging latent was free was an artifact of an undertrained model. At 30k
the lagging latent costs 17 points at t=1.0.

Implication: inference should start diffusion blocks only where `blt_next_starts_patch` says a
patch starts, finishing the open patch with ordinary AR steps first. The earlier "no large gap,
skip Step 6" advice is superseded.

#### 2. The `o_{i-1}` convention matters far more than previously measured

Cross-attention target `o_{i-1}` (paper) vs the last closed latent:

| t | delta | 95% CI |
|---|---|---|
| 1.00 | -0.0625 | [-0.0731, -0.0518] |
| 0.50 | -0.0955 | [-0.1128, -0.0782] |

The undertrained model measured -0.0056; the trained model measures -0.06 to -0.095. Using the
last closed latent instead of `o_{i-1}` costs 6-10 accuracy points.

#### 3. No leak, and the novel layout is not degraded

Train 0.2686 vs aligned-novel 0.3164 at t=1.0 (gap -0.0478), consistent with the leak fix
holding. The aligned novel layout now slightly exceeds the trained layout, which the weak model
never did.

## 6. Interpretation

The paper-exact loss works as intended and is not a marginal gain. Two conclusions from the
undertrained model are reversed: the aligned/midpatch gap is large, and the `o_{i-1}`
convention matters substantially. These are likely to hold at larger scales, but absolute
accuracies will rise.

## 7. Limitations and validity threats

- 32 windows / 2 sites for the layout table: `aligned`/`midpatch`/`train_lastlat` cells have
  `n = 152-256` per `t`, so their CIs are wide (+-0.07). The train rows (`n > 6000`) are
  precise. The midpatch and convention gaps are large enough to survive this.
- The 600k run has not been done yet; all quality numbers here are from 30k steps.
- The relative findings should hold, but absolute accuracies will rise.
- This report records measurement tables, not complete reproduction metadata.

## 8. Conclusion

At 30k steps the paper-exact loss reaches causal BPB 1.7536 and train-layout masked accuracy
0.518, a large improvement over the older run. The paper form makes clipping effectively
inert. The layout re-measurement reverses two earlier conclusions.

## 9. Follow-up work

A full 208-window run would tighten the layout CIs. The 600k run remains to be done. The
earlier "no large gap, skip Step 6" advice is superseded.

## Caveats

- 32 windows / 2 sites for the layout table: the `aligned`/`midpatch`/`train_lastlat` cells
  have `n = 152-256` per `t`, so their CIs are wide (+-0.07). The train rows (`n > 6000`) are
  precise. The midpatch and convention gaps are large enough to survive this; a full
  208-window run would tighten them.
- The 600k run has not been done yet; all quality numbers here are from 30k steps and will
  improve. The relative findings (leak closed, `o_{i-1}` wins, midpatch costly) should hold,
  but the absolute accuracies will rise.
