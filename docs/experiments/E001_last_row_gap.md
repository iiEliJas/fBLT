# Last-Row Training Gap: Why BLT-D Generation Failed Despite Near-Perfect Teacher-Forced Accuracy

## Metadata

- **Experiment ID:** E001
- **Title:** Last-row training gap and paper-conformance audit
- **Status:** Checked
- **Date or reporting period:** Not recorded in the historical report.
- **Author or contributors:** Not recorded in the historical report.
- **Related reports, issues, or papers:** [Forced-boundary patch closures](E004_forced_boundary_patches.md); Byte Latent Transformer; Fast Byte Latent Transformer.
- **Git commit or release:** Not recorded in the historical report.
- **Configurations and scripts:** Affected code is referenced in the report (`csrc/ops/patch_pool_cpu.c`, training loss, decoder wiring, global attention mask, BLT-D block-row attention). Full command lines, seeds, and checkpoint paths are not recorded.
- **Artifacts:** Validation summaries are recorded in the report. Checkpoint paths and raw result files are not recorded.

## 1. Research question

Why did a checkpoint with 97.76% teacher-forced top-1 accuracy produce repetitive garbage under
greedy generation?

## 2. Motivation and hypothesis

The forward pass was bit-identical between evaluation and generation, which ruled out an
inference bug and pointed at training. The investigation traced the cause through the training
loss, decoder cross-attention, and an audit of attention-masking mechanisms against both papers.

## 3. Experimental design

- Compared teacher-forced evaluation against greedy generation.
- Traced the forward pass and training loss.
- Audited attention-masking mechanisms against the Byte Latent Transformer and Fast Byte
  Latent Transformer specifications.
- Validated fixes with continuation runs and isolated single-patch tests.

## 4. Reproduction instructions

Reproduction details are incomplete in this historical report. It records affected source
locations and validation summaries, but not full command lines, seeds, checkpoint paths,
hardware, or raw result files.

## 5. Results

### Summary

A checkpoint scored 97.76% top-1 accuracy under teacher forcing, then produced repetitive
garbage under greedy generation. The forward pass was bit-identical between the eval and
generation code paths. Four real deviations were found and fixed; everything else already
matched.

### 2.1 Causal loss coverage

The training loop read exactly `window` bytes per example and computed loss over rows
`0..N-2`, so the last row of every window never got a gradient for 600k steps. Fixed by
reading `window + 1` bytes and supervising all `window` rows, matching Fast BLT Eq. 5's
full-sequence sum. Validated with a 300k-step continuation: last-row top-1 accuracy rose from
8-12% to 37-43%, uniformly across window sizes 8/64/256/512.

### 2.2 Decoder cross-attention, clean rows

`blt_patch_build_group_ids` (`csrc/ops/patch_pool_cpu.c:98-112`) assigned every byte its own
patch's group id unconditionally, with no final/non-final branch. Combined with the encoder's
bidirectional pooling within each patch (`patch_pool_cpu.c:28-35`), this let non-final rows,
roughly 81% of all rows, read their own target byte directly out of the patch representation
they were predicting from. Fixed to implement the paper's split rule (non-final -> previous
patch, final -> own patch) as an actual training-time change to the decoder wiring.

Validated with a 30k-step continuation: non-final accuracy rose from 16.94% to 66.42%, well
above a 2.00% unigram floor at both ends. The deficit at step 0 was genuine, not a
measurement artifact; a single-patch isolation test with `pre_flips = 0` ruled out
cross-contamination from other rows. This run had not converged when training stopped (loss
was still dropping at 30k steps), and 66.42% should not be read against the old ~98% figure as
a target still to close. That figure reflected the leaky task, not a legitimate ceiling for
the honest one.

### 2.3 Global attention mask: multi-document boundary scoping

The mask builder expects `num_docs - 1` boundary entries; the model-level API supplies
`num_docs` entries, and the bridging code forwarded the model-style array unchanged, so the
final document boundary was never read. For `num_docs = 2` this produced zero document
isolation. Inert in practice, since every production call site used `num_docs <= 1`. Reported
fixed.

### 2.4 BLT-D block-row self-attention: train/infer consistency

Training used strictly causal self-attention for block rows; inference used fully bidirectional
self-attention for the same rows. The paper specifies bidirectional attention within a block at
training time too (Fast BLT section 3.2.2, Fig. 5 caption). Inference already matched the
paper, but only by coincidence, through a convention that did not match what training was
doing. Reported fixed to bidirectional at both.

## 6. Interpretation

The symptom was caused by training, not inference. The fixes address causal loss coverage,
decoder cross-attention, global attention boundary scoping, and BLT-D train/inference
consistency. The remaining forced-boundary effect is a separate issue outside this
conformance scope.

## 7. Limitations and validity threats

- The 30k continuation had not converged.
- The original ~98% figure reflected the leaky task, not a legitimate ceiling.
- The global attention boundary issue was inert in production call sites.
- This report records validation summaries, not full reproduction metadata.

## 8. Conclusion

Investigation complete. Four confirmed deviations from the Byte Latent Transformer and Fast
Byte Latent Transformer papers were found in this implementation's attention and loss code
and are now fixed.

## 9. Follow-up work

The forced-boundary patch closure issue is related but outside paper-conformance scope; see
[E004_forced_boundary_patches.md](E004_forced_boundary_patches.md).

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
