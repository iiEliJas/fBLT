# Last-Row Training Gap: Why BLT-D Generation Failed Despite Near-Perfect Teacher-Forced Accuracy

**Status:** Root cause confirmed and fixed (see §3). A large gap remains at the last row of
every training window, and the two follow-up diagnostics in §7.3 didn't close it: they found a
real but small max-length-patch effect (~7pp within 2.8% of rows) and no evidence either way for
a truncation effect at the last row itself, since interior rows structurally can't test that.
The cross-attention rule means only the last byte of a patch ever reads that patch's own latent
(§7.3). Supervision imbalance (Hypothesis A, §6.1) is still the leading explanation but hasn't
been tested directly. Neither proposed fix (§7.1, §7.2) has been run yet. This document tracks
an open investigation and will keep changing.

## Summary

A BLT-D checkpoint (`tinystories_p4`, 600k steps) scored 97.76% top-1 accuracy under teacher
forcing, then produced repetitive garbage under greedy generation. The forward pass turned out
to be bit-identical between the two code paths, so this wasn't an inference bug. The real cause
was an off-by-one in the causal (`L_clean`) loss: training read exactly `window` bytes per
example and computed loss over rows `0..N-2`, so the last row of every window, the one predicting
the byte just past the edge, never got a gradient, for the whole run. Fixing this and continuing
training brought last-row accuracy up from ~8-12% to ~37-43%. Interior rows stayed at ~97-99%.
The gap is still large, and the rest of this document is about why.

## 1. Symptom

| Measurement | Result |
|---|---|
| Teacher-forced causal BPB (`causal_bpb`) | 0.1142 |
| Teacher-forced masked reconstruction accuracy | 100.00% (30672/30675) |
| Teacher-forced top-1 / top-5 / top-10 accuracy (`sanity_check`) | 97.76% / 99.86% / 99.96% |
| Greedy autoregressive generation | Coherent for a few real words, then collapses into repetition |
| Speculative draft acceptance rate | 0.0% |

A representative generation, prompted with real text:

```
Once upon a time, there was a cat
 that sat on a mat and it "Ot "Oo of of "Ot "Oo of of "Ot "Oe - - of "Oo of of "Ot "Okay"
```

The near-perfect training metrics made this look like an inference-side bug: a healthy model
shouldn't produce this from a real, in-distribution prompt.

## 2. How the Root Cause Was Isolated

Three results, together, ruled out an inference bug and pointed at training:

| Test | Result | Conclusion |
|---|---|---|
| `logits_compare`: same 512-byte input through `blt_model_forward()` vs. the manual encoder→global→decoder path used by `sanity_check` | Max abs. diff = 0.0, argmax differs at 0/512 positions | Forward pass is bit-identical, not a forward-computation bug |
| `last_row_acc`: accuracy at row `N-1` only, exact same forward path as `sanity_check` | 8-12% across window sizes 8/64/256/512, vs. 90-98% average over rows `0..N-2` | The gap is isolated to one specific row, not a general degradation |
| `--fixed-patches` (bypasses the entropy LM with fixed-stride patching) | Last-row accuracy unchanged (~28.5%) | The entropy patcher isn't the cause |

The collapse was flat across window lengths and landed on exactly one row, and the forward pass
was bit-exact. That combination only makes sense if the model never got a chance to learn this
specific prediction. It's a training problem, not a runtime one.

## 3. Root Cause: An Off-by-One in the Causal Loss

The training loop (`csrc/train_blt_d.c`, ~lines 338-340) reads exactly `window` bytes per
training example and computes the causal loss over rows `0..N-2` against targets `bytes[1..N-1]`.
The target for row `N-1` (the byte at absolute position `window`) is never read into the buffer,
so that row contributes no gradient. Every training window silently excluded its own final
prediction from supervision, for 600k steps.

This deviates from how the causal loss is specified in the paper this implementation follows. The
Fast Byte Latent Transformer paper (Kallini et al., 2026) defines the clean-sequence loss in
§3.2.3, Equation 5, as a sum over the **entire** clean sequence of length *N*:

$$\mathcal{L}_{\text{clean}}(\theta) = -\sum_{i=1}^{N} \log p_{\theta}(x_i \mid x_{<i})$$

Read literally, this includes `i = N`, the final byte of the sequence, predicted from everything
before it. Nothing in the equation excludes the last position; the implementation's `0..N-2` loop
bound was an off-by-one relative to this definition, not a simplification the paper suggests.

## 4. What the Papers Say (and Don't Say) About This

### 4.1 The diffusion loss already handles the equivalent boundary case

The same paper's block-construction procedure for the diffusion loss (§3.2.1, "Block
Construction") already plans for this exact situation: when a block runs past the end of the
sequence, it gets padded out to the full block length with a placeholder token rather than read
past the buffer. That's the boundary discipline the causal side of this implementation lacked.
`L_mask` pads out-of-range block positions correctly, but the loader feeding `L_clean` simply
never read far enough to have a target for its last row. (`L_mask` was checked separately and
confirmed to use unshifted targets with correct padding at boundaries; the bug was isolated to
`L_clean`.)

### 4.2 Neither paper discusses per-window supervision imbalance at the last row

Neither paper discusses last-position degradation, edge-of-window effects, or end-of-sequence
loss weighting. Both report aggregate metrics only: BPB, pass@1, BLEU (Fast BLT §4.2-4.3;
original BLT §5.2), averaged over full windows or full generations. A problem confined to one
row out of hundreds per window wouldn't show up in any of those numbers, even if it were present
in their own runs.

### 4.3 A related but distinct boundary effect the original paper *does* address

The original BLT paper does discuss a related boundary effect, but a different one: "entropy
drift," where the entropy model's patch-boundary decisions degrade over long, repetitive context
(§4.4, illustrated with an MMLU example where the same phrase gets patched very differently on
repeat occurrences). Their fix, resetting the entropy model's context at newlines and using the
approximate-monotonicity constraint instead of the global threshold, deals with patch-boundary
drift over a growing context, not loss coverage at the edge of a fixed training window. Easy to
conflate the two, since both are "boundary effects in BLT," but they're mechanistically distinct,
and fixing one doesn't touch the other.

### 4.4 Scale, not explicit handling, likely hides this in the published experiments

At the scale these papers train at, this exact imbalance, if it's even present in their own
pipelines, would be far less visible. The Fast BLT paper's 3B models train for 480,000 steps at
a batch size of roughly 2²⁰ tokens per step (Appendix A.2), on the 1-trillion-token BLT-1T corpus
(§4.1). Whether or not every window there also excludes its own last row from `L_clean`, the sheer
number of windows means the absolute count of last-row instances is enormous, and any residual
weakness at one relative position per window gets diluted to invisibility in an aggregate BPB or
pass@1 score. A small-scale fine-tune on a modest corpus, at 2.97M parameters, doesn't have that
luxury: the same relative imbalance produces a much smaller absolute count of last-row examples,
and the resulting weakness is large enough to dominate autoregressive generation, where every step
is effectively a "last row" prediction.

## 5. The Fix and Its Result

**Fix applied:** the training data loader was changed to read `window + 1` bytes per example
instead of `window`. The model still receives exactly `window` bytes as input: patching,
encoder, global model, decoder, and RoPE positions are unaffected. The extra byte extends the
`L_clean` target array to `window` entries, and the loss loop bound changed from `0..N-2` to
`0..N-1`, so every row in the window is now supervised. `L_mask` was left unchanged. Training
resumed from the existing 600k-step checkpoint rather than restarting, since interior-row
accuracy was already strong.

**Result**, after 300k additional steps with the corrected loss (`tinystories_p7`):

| Window | Last-row top-1 (pre-fix) | Last-row top-1 (post-fix) | Last-row mean CE (post-fix, nats) |
|---|---|---|---|
| 8   | 8.00%  | 37.00% | 3.60 |
| 64  | 9.00%  | 42.00% | 3.66 |
| 256 | 12.00% | 41.00% | 3.46 |
| 512 | 11.25% | 43.00% | 1.28 |

Interior-row accuracy stayed at ~97-99% throughout. The improvement is roughly uniform across
window lengths, not concentrated at `window = 512`, which rules out the fix only generalizing to
the one buffer length actually used during training.

Autoregressive generation improved without being fixed: recognizable word fragments show up now
instead of undifferentiated noise, but output still collapses into repetition after a short span.
A ~40% per-step top-1 rate, compounded autoregressively with nothing to correct it, would degrade
fast even in a genuinely well-trained model at that rate, so this reads as a real, partial
improvement rather than a failed fix, while confirming the gap to interior-row accuracy hasn't
closed.

## 6. Remaining Gap: Two Live Hypotheses

The off-by-one fix corrected the *presence* of a gradient at the last row, but not the *rate* at
which it's supervised relative to interior rows, and not the possibility that the last row draws
from a different input distribution altogether. Two hypotheses are under consideration for the
residual ~55-point gap between last-row and interior-row accuracy.

### 6.1 Hypothesis A: Per-window supervision imbalance

Every training window contributes exactly one last-row example against `window − 1` interior-row
examples, about 511:1 at `window = 512`. Fixed-window causal training makes this ratio
unavoidable, as both papers describe it (original BLT §4.3 fixes an 8k/16k-byte context per
dataset; Fast BLT §4.1 uses `window = 512` here), and neither paper's recipe corrects for it. At
the papers' scale this gets buried in sheer volume (§4.4); at this project's scale, only 300k
steps have given the last row any gradient at all, each contributing far fewer effective
last-row instances than interior-row ones. This also explains an otherwise odd detail in the
post-fix numbers: mean CE at short windows (3.5-3.7 nats) is roughly 3x worse than at
`window = 512` (1.28 nats) despite similar top-1, consistent with a model that guesses right
about as often, but with much less confidence, when it's had proportionally less exposure to
predicting past a short buffer's edge.

**Proposed fixes:** §7.1 and §7.2 below.

### 6.2 Hypothesis B: Truncated vs. naturally-closed patches

Entropy patching (original BLT §2.3; Fast BLT §2.1) places a boundary wherever the entropy
model's next-byte uncertainty crosses a threshold, by construction at genuinely hard-to-predict
transitions. The last byte of any training window is always the final byte of whichever patch
happens to be open when the buffer ends, whether or not the entropy model would have closed it
there on its own. If an arbitrarily cut-off patch pools into a systematically weaker latent than
one that closed naturally, the last row would be harder for a structural reason that has nothing
to do with gradient counts.

**Diagnostic result (§7.3 has now run, two stages):** neither supported nor refuted. Interior
rows of the buffer-boundary patch show no measurable deficit, but under the decoder's own
cross-attention rule, only the last byte of a patch ever reads that patch's own latent; every
other byte reads the *previous* patch's latent instead. So an interior-row test structurally
can't probe what this hypothesis is actually about. See §7.3 for the numbers and what a real
test would look like.

### 6.3 Considered and set aside

- **Learning-rate schedule during the 300k-step continuation**: considered, ruled out.
- **A hard ceiling from lack of right-context at the last position**: doesn't hold up, since
  every row of a causal prediction is equally blind to the right context, not just the last one.
  If this were the explanation, every row would show the same weakness.
- **The fix only working at the exact training window length (512)**: directly tested via the
  `last_row_acc` sweep across window sizes 8/64/256/512 (§5); the improvement is roughly uniform
  across all four, ruling this out.

## 7. Proposed Fixes (Not Yet Validated)

The two fixes target different mechanisms. §7.1 only changes *how much* gradient the existing
last-row examples contribute (Hypothesis A); it doesn't change what kinds of patches the model
sees. §7.2 addresses Hypothesis A too, by spreading last-row-style supervision across many
buffer lengths, and additionally exposes the model to more short, buffer-truncated patches, but
that second benefit only matters if a truncation-specific weakness actually exists at the last
row (Hypothesis B), which is still untested (§7.3). On present evidence §7.1 is the cheaper first
experiment, and §7.2's extra upside is unproven, not confirmed. Neither fix touches the one
effect §7.3 *did* measure: the deficit on max-length-capped patches, which would need a patcher
setting change, not a loss or data-loader change.

### 7.1 Loss upweighting for the last row

Multiply the last row's contribution to `L_clean` by a fixed factor (start near the per-window
imbalance ratio, ~511:1 at `window = 512`, and back off if interior accuracy regresses or
training destabilizes), so a single last-row example pulls proportionally more weight per step.
It's a one-line change to the loss, no data-loader changes needed, and the cheapest way to test
Hypothesis A in isolation. A clean result, last-row accuracy approaching the patch-final-row
baseline proposed in §7.3, would be strong evidence A dominates. A plateau well below it would
point to a last-row-specific representation gap that upweighting can't fix, which is the signal
to move to §7.2 and the truncation probe.

### 7.2 Variable-length training windows

Randomize the buffer length (or window start offset) sampled per training example instead of
always cutting at a fixed length. This spreads "predict past the edge" supervision across many
buffer lengths instead of piling it onto one relative position every time, and it directly
increases how often the model trains on genuinely short, buffer-truncated patches, unlike §7.1.
Whether that second part is worth anything depends on the last-row truncation probe in §7.3:
interior rows of buffer-boundary patches showed no deficit, but that test doesn't touch the last
row. This won't do anything for max-length-capped patches. It's a data-loader change on the same
scale as the off-by-one fix, and needs to be in the recipe from the start of any full retrain.

### 7.3 Diagnostics: Patch-Truncation Split Tests, Results

**Setup (both stages):** checkpoint `tinystories_p7` (900k steps total: 600k original plus 300k
with the corrected loss), evaluated on `data/tinystories/heldout.bin`, 50 windows / 25,550
interior-row predictions (`window = 512`, so 511 interior rows per window). Model: `embed=256`,
`hidden=512`, 2/6/2 encoder/global/decoder layers, `cross_attn=all`. Patcher: entropy-based
(`threshold_global=2.5`, `threshold_monotonic=1.0`, `max_patch_length=16`) via
`runs/entropylm/entropy_lm.fblt`. Interior rows are `0..N-2`; the true last row (`N-1`) is
excluded from every number below.

**Stage 1 (two-way split).** Rows split by how their governing patch closed: **natural** (a
genuine entropy trigger) versus **forced** (hit `max_patch_length`, or the buffer boundary).

| Patch closure type | Top-1 accuracy | Interior rows | Patches |
|---|---|---|---|
| Natural (entropy-triggered) | 98.55% | 24,722 | 4,760 |
| Forced (max length / buffer boundary) | 92.39% | 828 | 95 |
| Overall interior | 98.35% | 25,550 | 4,855 |

**Stage 2 (three-way split).** The forced bucket separated into its two
closure reasons:

| Bucket | Top-1 accuracy | 95% CI (rows independent) | Interior rows | Gap vs. natural |
|---|---|---|---|---|
| Natural (entropy-triggered) | 98.55% | 98.4-98.7% | 24,722 | (baseline) |
| Max-length capped (len = 16) | 91.67% | 89.4-93.5% | 720 | −6.88 pp |
| Buffer-boundary (final patch of each window) | 97.22% | 92.1-99.1% | 108 | −1.33 pp |

Buffer-boundary rows by the length of the truncated patch:

| Patch length | Top-1 accuracy | Interior rows | 95% CI |
|---|---|---|---|
| 1-2 bytes | 100.00% | 11 | 74-100% |
| 3-4 bytes | 100.00% | 37 | 91-100% |
| 5-8 bytes | 92.86% | 42 | 81-97.5% |
| 9-16 bytes | 100.00% | 18 | 82-100% |

**Consistency with Stage 1.** The two new buckets recombine exactly to the Stage 1 forced figure
(92.39%). 720 rows is exactly 45 capped patches × 16 rows, leaving 50 of the 95 forced patches as
buffer-boundary patches, one per window. So 87% of the Stage 1 forced rows were max-length rows,
matching the upper end of what the Stage 1 counts alone implied.

**Scale of the max-length effect.** Max-length rows are 2.82% of interior rows (720/25,550), so
the 6.88-point in-bucket deficit only costs 0.19pp of the aggregate interior mean. Bringing
capped patches up to the natural rate would move overall interior accuracy from 98.35% to about
98.54%. The ~7-point number describes accuracy *within* the bucket, not a drag on the interior
mean. Buffer-boundary rows are 0.4% of interior rows and contribute almost nothing to the
aggregate either way.

**Reading the buffer-boundary result with the right amount of confidence.** The bucket has 3
errors in 108 rows, all in the 5-8 byte sub-bucket. That's compatible with the natural-patch
error rate (chance of ≥3 errors in 108 rows at that rate: ~0.2) and looks somewhat better than
the max-length rate (chance of ≤3 errors at that rate: ~0.02, optimistically, since rows in the
same patch are correlated). The sub-buckets sitting at 100% are drawn from 11, 37, and 18 rows
out of at most 50 patches total, so their lower confidence bounds sit at 74-91%. The honest
summary is "no deficit detected," not "short patches are fine."

**Two things these tests cannot show.**

1. **They don't test Hypothesis B.** Under Fast BLT's decoder cross-attention pattern (§3.1.1),
   every byte in a patch except the last attends to the *previous* patch's latent; only the final
   byte reads its own patch's latent. Interior rows of the final patch are exactly the non-final
   bytes (the final byte is the excluded last row), so none of the 108 buffer-boundary rows ever
   touch the truncated patch's pooled representation. Only the last row does. High accuracy on
   this bucket says nothing about whether pooling over a 1-4 byte truncated patch produces a weak
   latent. Hypothesis B is *untested*, not refuted. The same limitation applies to any
   interior-row test of buffer-boundary patches. (This assumes the project's decoder mask follows
   §3.1.1; worth verifying directly in the implementation.)

2. **They don't explain why capped patches are harder.** Two explanations fit the data equally
   well:
   - *Depth-in-patch / stale latent.* 15 of the 16 rows in a capped patch read a latent from the
     previous patch, up to 15 bytes old, versus at most a few bytes for a typical natural patch
     (average 5.2 rows). Under the same mask logic, "the model pooled over content the entropy
     model couldn't parse" can't explain errors on most of these rows, since 15 of 16 never read
     their own patch's latent at all.
   - *Something about the content itself.* A 16-byte run with no trigger means the entropy model
     stayed below `threshold_global = 2.5` and never rose by more than `threshold_monotonic = 1.0`,
     not obviously "high entropy." The original BLT paper actually associates unusually long
     patches with repetitive, *low*-entropy content (§4.4, App. E). Entropy wasn't logged in this
     test, so which direction it points is unknown.

**Implications for the last-row gap.** The ~55-point gap (~40% last-row vs. ~97-99% interior) is
still unexplained by anything measured so far: the max-length deficit is ~7 points inside a
bucket that's 2.8% of rows, and buffer-boundary interior rows show no deficit at all. Hypothesis
A (§6.1) is still the leading candidate. The completed tests removed the only evidence pointing
toward B, but A hasn't been tested directly either, and B hasn't been tested at the last row at
all. Neither result says more last-row training *can't* close the gap. That's what §7.1 tests.

**`max_patch_length` note.** The Fast BLT paper trains with an average patch size of 4 bytes and
a max of **8** (§4.1); this project uses 16. A lower cap would shrink the number of capped,
deep-in-patch rows, but the model was trained on the 16-cap distribution, so changing it needs at
least a re-evaluation and realistically a retrain, and it increases patch count (more compute).
The upside for interior accuracy is bounded at ~0.2pp, and it wouldn't be expected to touch the
last-row gap, which isn't a max-length effect. Worth considering on its own merits, not as a fix
for this issue.

**Follow-up tests (not run yet)**

1. **Patch-final-row baseline** (re-analysis only, if per-row data are already logged). The last
   row is, by construction, a *patch-final* row: one that reads its own latent and predicts the
   byte right after a closure. Natural patch-final rows sit right before entropy-triggered
   boundaries, which are placed before hard-to-predict bytes on purpose, so they're inherently
   harder than the average interior row. The current comparison (~40% last-row vs. ~98%
   all-interior) mixes row types that shouldn't be compared directly. Report patch-final-row
   accuracy separately for natural and max-length patches, bucketed by patch length. That's the
   real baseline for the last row, the target §7.1 should be judged against.
2. **Last-row truncation probe**: the only test that actually exercises Hypothesis B. Run
   `last_row_acc`-style probes at many cut points within windows, bucket by the length of the
   open patch at the cut and by whether the cut lands on a natural boundary, and compare against
   the baseline from (1). If short-cut last-row accuracy sits meaningfully below that baseline, a
   truncation-specific representation gap is real, and §7.2 gains weight.
3. **Position and entropy breakdown inside capped patches.** Accuracy by position within the
   patch (1-16), and mean entropy per row/patch, compared against natural patches of similar
   length (12-15 bytes, where available). Errors concentrated at late positions point to a
   stale-latent effect; errors tracking entropy point to content. This diagnoses the max-length
   effect only; it doesn't bear on the last-row gap.







## 8. Summary Table

| Item | Status |
|---|---|
| Symptom: healthy teacher-forced eval, garbage greedy generation | Confirmed |
| Root cause: off-by-one excluding the last row from `L_clean` | Confirmed and fixed |
| Fix generalizes across window lengths (8/64/256/512); interior accuracy unaffected | Confirmed |
| Full recovery of last-row accuracy | **Not achieved** (residual gap, ~40% vs. ~97-99% interior) |
| Hypothesis A (per-window supervision imbalance) | Still the leading candidate; no direct test yet (§7.1) |
| Hypothesis B as originally stated (truncated patches → weak latent, tested via interior rows) | **Untested, not refuted** (interior rows never read the truncated patch's own latent, §7.3) |
| Max-length-capped patches are harder | **Measured**: 91.67% vs. 98.55% natural (−6.88pp, 720/25,550 rows); costs ~0.19pp aggregate, doesn't explain the ~55pp last-row gap |
| Interior rows of buffer-boundary patches are harder | **Not detected** (97.22%, 108 rows, 95% CI 92.1-99.1%); small sample, ≤50 patches |
| Loss upweighting (§7.1) | Proposed, untested (cheapest direct test of Hypothesis A) |
| Variable-length windows (§7.2) | Proposed, untested (the truncated-patch benefit is unsupported until the truncation probe runs) |
| `max_patch_length`: project uses 16, Fast BLT paper uses 8 | Deviation noted; upside bounded at ~0.2pp interior accuracy, unrelated to the last-row gap |

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
