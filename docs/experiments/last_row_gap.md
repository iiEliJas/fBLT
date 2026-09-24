# Last-Row Training Gap: Why BLT-D Generation Failed Despite Near-Perfect Teacher-Forced Accuracy

**Status: CLOSED.** Investigation complete. Four confirmed deviations from the Byte Latent
Transformer and Fast Byte Latent Transformer papers were found in this implementation's attention
and loss code and are now fixed.

## Summary

A checkpoint scored 97.76% top-1 accuracy under teacher forcing, then produced repetitive garbage
under greedy generation. The forward pass turned out to be bit-identical between the eval and
generation code paths, which ruled out an inference bug. Tracing the cause led through the
training loss, then the decoder's cross-attention, then a full audit of every attention-masking
mechanism against both papers' specifications. Four real deviations turned up and got fixed.
Everything else already matched.

## 1. Symptom

97.76% teacher-forced top-1 accuracy; greedy generation coherent for a few words, then collapsing
into repetition. `logits_compare` confirmed the eval and generation forward passes are
bit-identical (max abs. diff = 0.0), ruling out an inference bug and pointing at training.

## 2. What Was Fixed

### 2.1 Causal loss coverage

The training loop read exactly `window` bytes per example and computed loss over rows `0..N-2`,
so the last row of every window never got a gradient, for 600k steps. Fixed by reading `window +
1` bytes and supervising all `window` rows, matching Fast BLT Eq. 5's full-sequence sum. Validated
with a 300k-step continuation: last-row top-1 accuracy rose from 8-12% to 37-43%, uniformly across
window sizes 8/64/256/512.

### 2.2 Decoder cross-attention, clean rows

`blt_patch_build_group_ids` (`csrc/ops/patch_pool_cpu.c:98-112`) assigned every byte its own
patch's group id unconditionally, with no final/non-final branch. Combined with the encoder's
bidirectional pooling within each patch (`patch_pool_cpu.c:28-35`), this let non-final rows,
roughly 81% of all rows, read their own target byte directly out of the patch representation they
were predicting from. Fixed to implement the paper's split rule (non-final → previous patch, final
→ own patch) as an actual training-time change to the decoder wiring.

Validated with a 30k-step continuation: non-final accuracy rose from 16.94% to 66.42%, well above
a 2.00% unigram floor at both ends. The deficit at step 0 was genuine, not a measurement artifact;
a single-patch isolation test with `pre_flips = 0` ruled out cross-contamination from other rows.
This run hadn't converged when training stopped (loss was still dropping at 30k steps), and
66.42% shouldn't be read against the old ~98% figure as a target still to close. That figure
reflected the leaky task, not a legitimate ceiling for the honest one.

### 2.3 Global attention mask: multi-document boundary scoping

The mask builder expects `num_docs - 1` boundary entries; the model-level API supplies `num_docs`
entries, and the bridging code forwarded the model-style array unchanged, so the final document
boundary was never read. For `num_docs = 2` this produced zero document isolation. Inert in
practice, since every production call site used `num_docs ≤ 1`. Reported fixed.

### 2.4 BLT-D block-row self-attention: train/infer consistency

Training used strictly causal self-attention for block rows; inference used fully bidirectional
self-attention for the same rows. The paper specifies bidirectional attention within a block at
training time too (Fast BLT §3.2.2, Fig. 5 caption). Inference already matched the paper, but only
by coincidence, through a convention that didn't match what training was doing. Reported fixed to
bidirectional at both.

## 3. Related Finding, Outside Paper-Conformance Scope

Independent of everything above: a patch force-closed at a training window's edge, rather than
closed by a genuine entropy trigger, produces a measurably weaker final-byte prediction, driven by
closure type rather than patch length (a natural 1-byte patch scores 96.98%; a forced 1-byte patch
scores 8.33%). Neither paper discusses this. It's a side effect of training on fixed-length
windows, not a deviation from either paper's specification, so it sits outside this document's
conformance scope. It was not separately fixed. It showed a suggestive but unconfirmed improvement
(49.20% → 56.20%) alongside the §2.2 retrain and is worth re-measuring once a full retrain under
all corrected conventions is complete.

## References

```bibtex
@misc{pagnoni2024bytelatenttransformerpatches,
      title={Byte Latent Transformer: Patches Scale Better Than Tokens},
      author={Artidoro Pagnoni and Ramakanth Pasunuru and Pedro Rodriguez and John Nguyen and
              Benjamin Muller and Margaret Li and Chunting Zhou and Lili Yu and Jason Weston and
              Luke Zettlemoyer and Gargi Ghosh and Mike Lewis and Ari Holtzman and Srinivasan Iyer},
      year={2024},
      eprint={2412.09871},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      doi={10.48550/arXiv.2412.09871}
}

@misc{kallini2026fastbytelatenttransformer,
      title={Fast Byte Latent Transformer},
      author={Julie Kallini and Artidoro Pagnoni and Tomasz Limisiewicz and Gargi Ghosh and
              Luke Zettlemoyer and Christopher Potts and Xiaochuang Han and Srinivasan Iyer},
      year={2026},
      eprint={2605.08044},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
