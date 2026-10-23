# 128-Patch Cap

## Metadata

- **Experiment ID:** E005
- **Title:** 128-patch cap investigation
- **Status:** Exploratory
- **Date or reporting period:** Not recorded in the historical report.
- **Author or contributors:** Not recorded in the historical report.
- **Related reports, issues, or papers:** [Last-row training gap](E001_last_row_gap.md); Byte Latent Transformer; Fast Byte Latent Transformer.
- **Git commit or release:** Not recorded in the historical report.
- **Configurations and scripts:** No measurements recorded. References `BLT_MAX_PATCHES`, the patcher, and the BLT-D decoder sequence budget.
- **Artifacts:** None recorded.

## 1. Research question

Is the 128-patch cap justified, and what would be required to raise it?

## 2. Motivation and hypothesis

The cap was added because BLT-D did not support more patches at the time. It is an
implementation limit, not a limit from either paper.

## 3. Experimental design

No experiment has been run yet. The proposed design is to separate patcher storage capacity
from the BLT-D sequence limit: the patcher should represent any valid segmentation, while
the trainer rejects or handles windows whose expanded block sequence exceeds the model
budget.

## 4. Reproduction instructions

No measurements or reproduction commands are recorded. This document describes an open
experiment.

## 5. Results

No measurements recorded.

## 6. Interpretation

With the TinyStories config (`window: 512`, entropy patching, `block-size: 4`), a window can
need more than 128 patches. The patcher warns when its output array is full. In the training
loop, the saturated window is skipped rather than trained with an incomplete patch list, so
the run continues but that optimizer step is wasted.

The cap is close to the current BLT-D decoder budget. BLT-D adds `4 * (M - 1)` block rows to
the clean sequence, so a larger patch count also increases decoder sequence length. A patcher
cap of 512 would therefore need a matching change to the model limit or block handling;
simply replacing `128` with `512` is not enough.

## 7. Limitations and validity threats

- No measurements have been taken.
- Patch count, skipped-window rate, memory use, throughput, and BPB are all unmeasured.
- Neither paper specifies a 128-patch maximum.

## 8. Conclusion

The 128-patch cap is an implementation limit, not a paper limit. Raising it requires
separating patcher capacity from the BLT-D sequence budget and measuring the effects.

## 9. Follow-up work

Run the proposed experiment: measure patch-count distribution, skipped-window rate, memory
use, throughput, and BPB before choosing a new cap.

## Note

Neither paper specifies a 128-patch maximum. They describe variable patch counts and use
average patch size to control compute. The 512-byte attention window mentioned in the papers
is a separate setting from the number of patches.
