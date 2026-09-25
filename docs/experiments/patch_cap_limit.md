# 128-Patch Cap

**Status: OPEN.** I added the 128-patch cap because BLT-D did not support more patches at the time. It is an implementation limit, not a limit from either paper, so raising it and measuring the effect is still an open experiment.

## What happened

With the TinyStories config (`window: 512`, entropy patching, `block-size: 4`), a window eventually needed more than 128 patches. The patcher warns when its output array is full. In the training loop, the saturated window is skipped rather than trained with an incomplete patch list, so the run continues but that optimizer step is wasted.

The cap is close to the current BLT-D decoder budget. BLT-D adds `4 * (M - 1)` block rows to the clean sequence, so a larger patch count also increases the decoder sequence length. A patcher cap of 512 would therefore need a matching change to the model limit or block handling; simply replacing `128` with `512` is not enough.

## Open experiment

The next useful experiment is to separate the patcher storage capacity from the BLT-D sequence limit. The patcher should be able to represent any valid segmentation of the window, while the trainer rejects or handles windows whose expanded block sequence exceeds the model budget. Measure the patch-count distribution, skipped-window rate, memory use, throughput, and BPB before choosing a new cap.

Neither paper specifies a 128-patch maximum. They describe variable patch counts and use the average patch size to control compute. The 512-byte attention window mentioned in the papers is a separate setting from the number of patches.