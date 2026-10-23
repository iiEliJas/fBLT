# Forced-Boundary Patch Closures Weaken Final-Byte Prediction

## Metadata

- **Experiment ID:** E004
- **Title:** Forced-boundary patch closures weaken final-byte prediction
- **Status:** Checked
- **Date or reporting period:** Not recorded in the historical report.
- **Author or contributors:** Not recorded in the historical report.
- **Related reports, issues, or papers:** [Last-row training gap](E001_last_row_gap.md); Byte Latent Transformer; Fast Byte Latent Transformer.
- **Git commit or release:** Not recorded in the historical report.
- **Configurations and scripts:** References `sanity_check`, `masked_acc`, and patch-closure measurements. Exact commands, seeds, checkpoint paths, and evaluation scripts are not recorded.
- **Artifacts:** Measurement tables are recorded in the report. Checkpoint names and raw result files are not recorded.

## 1. Research question

Does force-closing a patch at a training-window boundary weaken final-byte prediction, and is
the effect driven by closure type rather than patch length?

## 2. Motivation and hypothesis

This surfaced during the last-row investigation. Row `N-1` is always the final byte of whichever
patch is open when the window buffer ends, whether or not the entropy patcher would have closed
it there. The hypothesis is that a patch closed that way pools an arbitrary span, producing a
weaker representation for final-byte prediction.

## 3. Experimental design

- Compared natural entropy closures with forced boundary closures at matched patch lengths.
- Tracked boundary-final accuracy across four checkpoints with different training histories.
- Compared final-row and non-final-row accuracy by closure type.
- Measured a length breakdown across all 256,000 rows.
- Ran `sanity_check` over 50 windows and 25,550 held-out bytes for the completed run.

## 4. Reproduction instructions

Reproduction details are incomplete in this historical report. It records measurement tables and
checkpoint histories, but not exact commands, seeds, checkpoint paths, hardware, or evaluation
scripts.

## 5. Results

### Origin

Same row as [E001_last_row_gap.md](E001_last_row_gap.md) section 2.1. The off-by-one loop bound excluded
the row from the loss entirely; once fixed, it trains like every other row, but its final-byte
prediction is still measurably weaker.

### Length-matched closure comparison

| Patch length | Natural closure | Forced boundary closure | Gap |
|---|---|---|---|
| 1 byte | 96.98% (n=995) | 8.33% (n=96) | 88.65pp |
| 1-2 | 98.49% (n=4,897) | 29.81% (n=208) | 68.68pp |
| 3-4 | 99.76% (n=19,567) | 61.11% (n=144) | 38.65pp |
| 5-8 | 99.05% (n=17,891) | 58.72% (n=109) | 40.33pp |
| 9-16 | 96.16% (n=5,362) | 82.05% (n=39) | 14.11pp |

Length-matched, the gap does not shrink. Closure type drives this, not length.

### Tracking across checkpoints

| Checkpoint | Training history | bnd-final accuracy | 95% CI | n |
|---|---|---|---|---|
| Original (leaky decoder convention) | 900k steps | 49.20% | 44.8-53.6% | 500 |
| Continuation, post cross-attention fix | 900k leaky steps + 30k continuation | 56.20% | 51.8-61.5% | 500 |
| From-scratch, both fixes present | 300k of planned 600k steps | 42.80% | 38.5-47.2% | 500 |
| Same run, completed | +300k continuation, 600k total | 50.00% | 45.6-54.4% | 500 |

All four are measured on the same 500-window sample. The from-scratch run gained 7.20pp between
its own two measurements.

### Closure-type comparison on the completed run

| Closure | Final-row acc | Non-final-row acc | Gap (final - nonfinal) | Final-row delta vs 300k |
|---|---|---|---|---|
| natural | 99.38% (47,419/47,717) | 69.53% (138,047/198,546) | +29.85pp | +0.17pp |
| max-length | 88.39% (434/491) | 73.84% (5,438/7,365) | +14.56pp | +2.44pp |
| boundary | 50.00% (250/500) | 61.40% (848/1,381) | -11.40pp | +7.20pp |

Natural and max-length closures show the expected sign. Boundary closures invert it.

### Length breakdown on the completed run

| Patch length | Final-row acc | Non-final-row acc | Gap |
|---|---|---|---|
| 1-2 | 95.61% (4,881/5,105) | 46.34% (1,860/4,014) | +49.27pp |
| 3-4 | 99.52% (19,617/19,711) | 71.39% (37,089/51,954) | +28.13pp |
| 5-8 | 99.03% (17,826/18,000) | 67.94% (61,422/90,403) | +31.09pp |
| 9-16 | 98.08% (5,779/5,892) | 72.16% (43,962/60,921) | +25.92pp |

### Sanity check on the completed run

| Metric | Value |
|---|---|
| Mean CE | 0.7952 nats/byte |
| Mean BPB | 1.1473 bits/byte |
| Top-1 | 75.04% |
| Top-5 | 93.20% |
| Top-10 | 97.42% |

## 6. Interpretation

Under the corrected decoder convention, a final-byte row reads its own patch's latent. For
natural and max-length patches, "own patch" is a good patch, so it wins. For boundary patches,
"own patch" is the arbitrary forced closure, so reading it loses to the preceding ordinary patch.

The effect appears in all four checkpoints, across leaky, corrected, half-trained, and
fully-trained histories, at 42.80% to 56.20%. The boundary-final number improved 7.20pp in
stage 2, about twice the interior-row improvement, but n=500 leaves the ratio suggestive
rather than established.

## 7. Limitations and validity threats

- n=500 for boundary-final rows, so intervals are wide.
- The 7.20pp stage-2 gain is confounded by simultaneous learning-rate and mask-scale changes.
- The length breakdown restricted to boundary-closed patches has not been run.
- The proposed sliding-window fix is an idea, not an implemented or tested change.
- This report records measurement tables, not complete reproduction metadata.

## 8. Conclusion

Not fixed, and there is no code-level deviation to point at, since neither paper specifies how
to handle a window boundary landing mid-patch. The effect is a byproduct of chopping training
data into fixed-length windows for causal training.

## 9. Follow-up work

Still open: the length breakdown restricted to boundary-closed patches, and a controlled run
that attributes the stage-2 gain to duration rather than schedule changes. A possible future
fix is overlapping or sliding training windows.

## Why neither paper mentions it

Both papers report aggregate metrics over much larger training runs. A weakness confined to one
relative position per training window, diluted across a dataset that size, would not move an
aggregate enough to notice, even if present in their own runs.

## Status

Not fixed. The effect is not a training-duration artifact; doubling the step budget moved the
number 7.20pp but did not close the deficit. What holds is that the effect appears in every
history measured.
