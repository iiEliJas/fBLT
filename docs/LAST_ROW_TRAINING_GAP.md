# Last-Row Training Gap: Why BLT-D Generation Failed Despite Near-Perfect Teacher-Forced Accuracy

**Status:** Two independent issues have been found in this implementation relative to what the
papers it follows specify.

**Root Cause #1** (§3): an off-by-one in the causal (`L_clean`) loss excluded the last row of
every training window from supervision, for 600k steps. **Confirmed and fixed.** Continuing
training with the fix raised last-row accuracy from ~8-12% to ~37-43%, leaving a large residual
gap to interior-row accuracy (~97-99%).

Extensive diagnosis of that residual gap (§7) found a real, independent, but comparatively small
effect: patches force-closed at a training window's edge (rather than closed by a genuine entropy
trigger) produce a measurably weaker final-byte prediction, worsening with shorter truncation
(§7.4, §7.6). Attempts to fix this by upweighting the last row's loss term recovered only a small
fraction of the gap at real cost to other rows (§7.1). Investigating why led to a second,
considerably larger issue.

**Root Cause #2 — candidate, mechanism confirmed, magnitude not yet measured** (§9): the decoder's
cross-attention indexing does not implement the rule either paper specifies. Both BLT and Fast BLT
say a patch's non-final bytes should attend the **previous**, already-complete patch's latent;
only a patch's final byte should attend its **own** latent. This implementation has every row —
final or not — attend its own patch's latent, unconditionally, in training and in all three
generation paths alike (§9, code cited directly against the paper's text). Because the encoder
pools each patch bidirectionally over its full span, this means most training-time predictions
have direct access to their own target byte through the patch representation they're reading from.
Roughly 81% of every row measured anywhere in this document prior to §8 — every non-final row in
every table in §7.3, §7.4, §7.6 — falls into this category. Final-byte rows are unaffected (they
were never leaking) and their findings stand. A from-scratch replication of this convention under
teacher forcing, using the true generation-time segmentation, produced 14.66%-69.77% accuracy
depending on how much of the current patch had accumulated (§8.3) — far below anything reported as
"interior accuracy" earlier in this document, because those earlier numbers were mostly measuring
a different, easier task.

This document tracks an open investigation and will keep changing. This version restructures
around the §9 finding and folds in everything found since the last update: the §7.1 upweighting
result, the Phase 0 offset/distribution investigation (§7.5), the length-matched closure-type
comparison (§7.6), and the full generation-path trace that led to §9 (§8).

## Summary

A BLT-D checkpoint (`tinystories_p4`, 600k steps) scored 97.76% top-1 accuracy under teacher
forcing, then produced repetitive garbage under greedy generation. The forward pass turned out to
be bit-identical between the two code paths, so this wasn't an inference bug. The first cause found
was an off-by-one in the causal (`L_clean`) loss: training read exactly `window` bytes per example
and computed loss over rows `0..N-2`, so the last row of every window, the one predicting the byte
just past the edge, never got a gradient, for the whole run. Fixing this and continuing training
brought last-row accuracy up from ~8-12% to ~37-43%. A large gap to interior-row accuracy (~97-99%)
remained, and a long diagnostic chain (§7-§8) tracing that gap eventually found that "interior
accuracy" itself was not measuring what it appeared to: the decoder's cross-attention lets most
rows read their own patch's completed representation, which — because the encoder pools that
representation bidirectionally — usually already contains the byte the row is being asked to
predict. This is a deviation from both papers' specified decoder attention rule, present
identically in training and in every generation path, and it is the most likely explanation left
standing for why a model that looked nearly perfect under the original measurement collapses so
quickly once it has to generate for real.

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

## 2. How Root Cause #1 Was Isolated

Three results, together, ruled out an inference bug and pointed at training:

| Test | Result | Conclusion |
|---|---|---|
| `logits_compare`: same 512-byte input through `blt_model_forward()` vs. the manual encoder→global→decoder path used by `sanity_check` | Max abs. diff = 0.0, argmax differs at 0/512 positions | Forward pass is bit-identical, not a forward-computation bug |
| `last_row_acc`: accuracy at row `N-1` only, exact same forward path as `sanity_check` | 8-12% across window sizes 8/64/256/512, vs. 90-98% average over rows `0..N-2` | The gap is isolated to one specific row, not a general degradation |
| `--fixed-patches` (bypasses the entropy LM with fixed-stride patching) | Last-row accuracy unchanged (~28.5%) | The entropy patcher isn't the cause |

The collapse was flat across window lengths and landed on exactly one row, and the forward pass
was bit-exact. That combination only makes sense if the model never got a chance to learn this
specific prediction. It's a training problem, not a runtime one.

*(Note added in this revision: the 90-98% comparison figure in the middle row above is itself an
interior-row average, and §9 later finds most interior rows measure a leaky task. This doesn't
change the conclusion of this section — the last row genuinely got zero gradient, a real bug,
correctly diagnosed and fixed below — but it means the size of the gap this table first
surfaced was never quite comparing like with like. See §9.)*

## 3. Root Cause #1: An Off-by-One in the Causal Loss

The training loop (`csrc/train_blt_d.c`, ~lines 338-340) reads exactly `window` bytes per training
example and computes the causal loss over rows `0..N-2` against targets `bytes[1..N-1]`. The target
for row `N-1` (the byte at absolute position `window`) is never read into the buffer, so that row
contributes no gradient. Every training window silently excluded its own final prediction from
supervision, for 600k steps.

This deviates from how the causal loss is specified in the paper this implementation follows. The
Fast Byte Latent Transformer paper (Kallini et al., 2026) defines the clean-sequence loss in
§3.2.3, Equation 5, as a sum over the **entire** clean sequence of length *N*:

$$\mathcal{L}_{\text{clean}}(\theta) = -\sum_{i=1}^{N} \log p_{\theta}(x_i \mid x_{<i})$$

Read literally, this includes `i = N`, the final byte of the sequence, predicted from everything
before it. Nothing in the equation excludes the last position; the implementation's `0..N-2` loop
bound was an off-by-one relative to this definition, not a simplification the paper suggests.

## 4. What the Papers Say (and Don't Say) About Root Cause #1

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

*(Note added in this revision: "interior-row accuracy" in the table above and throughout §6-§7 is
the same leaky-convention measurement flagged in §9. This doesn't undermine the fix itself — it
targeted a literal, unambiguous zero-gradient bug and measurably improved last-row accuracy at
every window length — but the ~97-99% figure it's compared against is now known to be inflated for
most of the rows that make it up. The true remaining gap, once §9 is resolved, may look different
from "40% vs. 97-99%.")*

## 6. Remaining Gap After Root Cause #1: Two Original Hypotheses

The off-by-one fix corrected the *presence* of a gradient at the last row, but not the *rate* at
which it's supervised relative to interior rows, and not the possibility that the last row draws
from a different input distribution altogether. Two hypotheses were considered for the residual
gap between last-row accuracy (measured ~49% in §7.4; ~40% in §5) and interior-row accuracy
(~97-99%).

### 6.1 Hypothesis A: Per-window supervision imbalance — tested, minor effect only

Every training window contributes exactly one last-row example against `window − 1` interior-row
examples, about 511:1 at `window = 512`. Fixed-window causal training makes this ratio
unavoidable, as both papers describe it (original BLT §4.3 fixes an 8k/16k-byte context per
dataset; Fast BLT §4.1 uses `window = 512` here), and neither paper's recipe corrects for it.

**Status:** directly tested via loss upweighting (§7.1). The control-adjusted effect was small
(+3.8pp on the target metric, at real cost to other rows) and a follow-up distribution analysis
(§7.5) found the exposure-imbalance framing doesn't even hold in the direction this hypothesis
predicts — the shortest, supposedly most-underexposed truncation length is also the *most common*
one in training, not the rarest. Hypothesis A is not the dominant driver of the post-fix gap.

### 6.2 Hypothesis B: Truncated vs. naturally-closed patches — confirmed, refined, and small relative to §9

Entropy patching (original BLT §2.3; Fast BLT §2.1) places a boundary wherever the entropy
model's next-byte uncertainty crosses a threshold, by construction at genuinely hard-to-predict
transitions. The last byte of any training window is always the final byte of whichever patch
happens to be open when the buffer ends, whether or not the entropy model would have closed it
there on its own. If an arbitrarily cut-off patch pools into a systematically weaker latent than
one that closed naturally, the last row would be harder for a structural reason that has nothing
to do with gradient counts.

**Status:** confirmed directly (§7.4: boundary-final rows collapse to 49.20% vs. 97.03% for the
same patch's non-final rows) and refined (§7.6: the effect is driven by *closure type*, not patch
*length* — natural patches as short as 1 byte score 96.98%, while forced-boundary patches of the
same length score 8.33%). This is real and independent of §9's finding, since final-byte rows are
never subject to the leak §9 describes under either convention. It is, however, small relative to
§9 in scope: it affects only the one row per training window that happens to land on a forced
closure, whereas §9 affects the majority of all rows, everywhere.

### 6.3 Considered and set aside

- **Learning-rate schedule during the 300k-step continuation**: considered, ruled out.
- **A hard ceiling from lack of right-context at the last position**: doesn't hold up, since
  every row of a causal prediction is equally blind to the right context, not just the last one.
  If this were the explanation, every row would show the same weakness.
- **The fix only working at the exact training window length (512)**: directly tested via the
  `last_row_acc` sweep across window sizes 8/64/256/512 (§5); the improvement is roughly uniform
  across all four, ruling this out.

### 6.4 A third mechanism, found while investigating A and B

Neither A nor B, even combined, accounts for why *generation* — as opposed to training-window
measurement — degrades as fast and as early as it does. Pursuing that question (§8) surfaced a
mechanism outside either hypothesis: the decoder's cross-attention convention itself, applied
identically to every row regardless of finality. See §9.

## 7. Diagnostics and Fix Attempts for the Post-Fix Gap

### 7.1 Loss Upweighting for the Last Row — Tested

**Design.** Multiply the last row's contribution to `L_clean` by ~511:1 (the per-window
imbalance ratio at `window = 512`), so a single last-row example pulls proportionally more weight
per step. A one-line loss change, no data-loader changes, the cheapest direct test of Hypothesis A.

**Result (×511, 20k continuation steps from the `p7` checkpoint):**

| Bucket (patch length) | n | Baseline | ×511 | Δ |
|---|---|---|---|---|
| 1-2 | 208 | 29.81% | 35.58% | +5.77pp |
| 3-4 | 144 | 61.11% | 67.36% | +6.25pp |
| 5-8 | 109 | 58.72% | 64.22% | +5.50pp |
| 9-16 | 39 ⚠small-n | 82.05% | 92.31% | +10.26pp |
| **Overall bnd-final** | **500** | **49.20%** | **55.40%** | **+6.20pp** |

Collateral, same run: natural-final 98.96%→97.25% (−1.71pp), natural-nonfinal 98.54%→95.50%
(−3.04pp), max-final 84.32%→83.50% (−0.82pp), max-nonfinal 91.38%→81.82% (−9.56pp), boundary-
nonfinal 97.03%→91.53% (−5.50pp). Total panel accuracy: 98.28%→95.31%.

**Matched control (same 20k steps, `--last-row-scale 1.0`, same LR schedule, from the same `p7`
checkpoint):** boundary-final overall 49.20%→51.60% (training alone is mildly positive). The
control does *not* reproduce any of the ×511 collateral damage — e.g. max-nonfinal
91.38%→92.07% under the control vs. →81.82% under ×511 — cleanly attributing the collateral cost
to the gradient skew itself, not to additional training.

**Control-adjusted result:** ×511 minus control = **+3.80pp** on the target metric (49.20% base):

| Bucket | Baseline | ×511 | Control | ×511 − control |
|---|---|---|---|---|
| 1-2 | 29.81% | 35.58% | 31.25% | +4.33pp |
| 3-4 | 61.11% | 67.36% | 66.67% | +0.69pp |
| 5-8 | 58.72% | 64.22% | 57.80% | +6.42pp |
| 9-16 ⚠small-n | 82.05% | 92.31% | 87.18% | +5.13pp |
| **Overall** | **49.20%** | **55.40%** | **51.60%** | **+3.80pp** |

**Verdict.** +3.8pp control-adjusted, against a ~47-49pp gap, is a real but small effect, and it
comes at a disproportionate cost elsewhere (roughly 250 rows degraded per 1 row gained on the
256k-row panel). This is the plateau §7.2's design anticipated as the signal to look elsewhere: not
enough of the gap is supervision-rate-driven for upweighting alone to close it. An intermediate
scale (8-32×) would very likely trade less collateral for proportionally less gain along roughly
the same line, rather than escaping it — not judged worth a dedicated run given what's in §9.

### 7.2 Variable-Length Training Windows — Proposed, Rationale Since Revised Twice

**Original design:** randomize the buffer length or window start offset sampled per training
example, to spread "predict past the edge" supervision across many buffer lengths and increase
exposure to short, genuinely truncated patches.

**First revision (§7.5):** the offset/length-randomization framing assumed truncation exposure was
fixed or narrow. Phase 0 found window start offsets *are* fixed (deterministic, non-overlapping
stride), but the resulting truncation-length distribution at the cut is already broad and
content-driven — randomizing offsets wouldn't change it. Worse, the distribution is *inverted*
relative to what an exposure-based fix would want: the worst-performing bucket (1-2 bytes) is also
the most common one.

**Second revision (§7.6):** the length-matched comparison then showed length isn't the operative
variable at all — a natural 1-byte patch scores 96.98%, a forced 1-byte patch scores 8.33%. A fix
that changes *which lengths* get truncated (any offset-randomization variant) doesn't address a
problem that isn't about length. What might still be worth testing is a fix that changes what
*kind* of example a given stretch of content becomes — e.g. overlapping/sliding windows, so
content that's arbitrarily cut in one window appears naturally-closed or interior in another,
diluting forced-cut supervision for the same content rather than reshuffling cut lengths. Not
implemented or tested. Deprioritized below §9's diagnostic, since even a successful version of this
fix would only address the §6.2/§7.4/§7.6 effect (one row per window), not the much larger §9
effect.

### 7.3 Diagnostics: Patch-Truncation Split Tests (Original Results, Unchanged)

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

**Stage 2 (three-way split).** The forced bucket separated into its two closure reasons:

| Bucket | Top-1 accuracy | 95% CI | Interior rows | Gap vs. natural |
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

**Scale of the max-length effect.** Max-length rows are 2.82% of interior rows (720/25,550), so
the 6.88-point in-bucket deficit only costs 0.19pp of the aggregate interior mean. This remains a
real, separate, minor effect: worth considering for the patcher's `max_patch_length` setting on its
own merits, not as a fix for anything else in this document.

**What this stage could not show, and why.** Under the paper's specified decoder rule, only a
patch's final byte reads its own latent — interior rows never do. The rows measured here are all
interior (`0..N-2`), so under the paper's rule this test structurally cannot probe whether a
truncated patch's own latent is weak. It was only after §9 confirmed the implementation does *not*
follow that rule that this limitation became moot for a different reason: interior rows here are
not blind to their own patch either way, in this codebase, and most of them are additionally
leaking their own target byte (§9). §7.4 below tested the one row (the true last row) that both
conventions agree reads its own patch's latent, and found the real deficit.

### 7.4 Patch-Final-Row Baseline (Confirmed)

**What this tested.** The last row (`N-1`) is by construction a *patch-final* row: the final byte
of the window's boundary (truncated) patch, and — under either the paper's rule or this repo's
rule — the one byte that reads that patch's own latent. This test measures patch-final-row
accuracy separately for all three closure types, bucketed by patch length, against the same rows'
non-final counterparts.

**Setup.** Diagnostic `patch_final_split` (`csrc/tools/patch_final_split.c`), analyzer
`fblt/scripts/analyze_patch_final.py`. Same model/corpus/patcher as §7.3: 500 windows / 256,000
rows. Validated against §7.3 byte-for-byte on a 50-window reproduction (98.55/91.67/97.22/98.35%,
identical) and against `last_row_acc` independently (48.00% = 24/50, matching this tool's 24/50
boundary-final figure on the same windows).

**Boundary (truncated) patch — the last row:**

| Patch length | Final-row acc | n | Non-final-row acc | n |
|---|---|---|---|---|
| 1-2  | 29.81% (24.00-36.34) | 208 | 100.00% (96.68-100.00) | 112 |
| 3-4  | 61.11% (52.96-68.69) | 144 | 97.44% (95.20-98.65) | 351 |
| 5-8  | 58.72% (49.33-67.51) | 109 | 97.28% (95.56-98.34) | 551 |
| 9-16 | 82.05% (67.33-91.02) ⚠small-n | 39 | 95.37% (92.71-97.09) | 367 |
| **all** | **49.20% (44.84-53.57)** | **500** | **97.03% (96.00-97.80)** | **1,381** |

**Natural and max-capped patches — patch-final rows are not inherently harder:**

| Closure | Final-row acc | n | Non-final-row acc | n | Gap |
|---|---|---|---|---|---|
| natural | 98.96% (98.87-99.05) | 47,717 | 98.54% (98.48-98.59) | 198,546 | −0.42pp (final higher) |
| max (len 16) | 84.32% (80.84-87.27) | 491 | 91.38% (90.72-92.00) | 7,365 | +7.06pp |
| boundary | 49.20% (44.84-53.57) | 500 | 97.03% (96.00-97.80) | 1,381 | +47.83pp |

**What this established.** The ~49% last row is a boundary-truncation effect, not a generic
patch-final-row effect. Natural patch-final rows are even slightly *better* than non-final natural
rows. The deficit is confined to the final byte of a *forced*-closure patch, worsening with shorter
truncation.

### 7.5 Phase 0: Window Offset Sourcing & Truncation-Length Distribution

**Question.** Does randomizing where training windows start (per §7.2's original design) change
what the model is exposed to, and would it plausibly help?

**Mechanism, confirmed by direct code trace.** Training windows are fixed-stride, non-overlapping,
deterministic tiles of the corpus (`train_blt_d.c:298,331,338-339`: `w = step % num_windows`,
`text = corpus + w * window`). No offset, shuffle, or jitter mechanism exists anywhere in
`train_args.c`; the RNG is used only for the corruption mask and diffusion timestep draws. Every
epoch revisits the same cells with the same segmentation. Training and the `patch_final_split`
measurement tool run the identical segmentation code path (`entropy_segment`,
`train_eval.c:135-167`, confirmed against `patcher.c:53-93`'s unconditional final-patch emission).

**Empirical truncation-length distribution** (500 `train.bin` windows, same entropy LM and
segmentation as training):

| Bucket | Training windows | Fraction |
|---|---|---|
| 1-2 | 206 | 41.2% |
| 3-4 | 143 | 28.6% |
| 5-8 | 108 | 21.6% |
| 9-16 | 43 | 8.6% |
| 17+ | 0 | 0% |

Mean truncation length 3.90 bytes. The heldout set (from the `patch_final_500w` control data) is
statistically indistinguishable: 208/144/109/39, mean 3.76.

**Verdict.** The truncation-length distribution is already broad, not fixed or narrow — the
premise behind offset-randomization as originally conceived doesn't hold. Worse for that design:
the distribution is **inverted** relative to what an exposure-fix would predict — the worst-
performing bucket (1-2 bytes, §7.4: 29.81%/31.25%) is the *most* common (41.2% of windows); the
best-performing bucket (9-16, 82-87%) is the *rarest* (8.6%). If undertraining on short truncations
explained the gap, the most-trained-on bucket should perform best, not worst. It doesn't.
Randomizing start offsets would change alignment/cold-start diversity, not truncation-width
diversity — a different, more weakly-motivated intervention than what was originally proposed.

### 7.6 Length-Matched Natural vs. Boundary Comparison

**Question.** Is a truncated patch hard because it's *short*, or because it was *forced closed*
rather than entropy-triggered? §7.3/§7.4 couldn't separate these — natural patches of matching
short length weren't in the comparison.

**Result** (re-analysis of the existing 500-window `patch_final_split` dump, no new run):

| Bucket | nat-final | bnd-final | Gap |
|---|---|---|---|
| 1-2 | 98.49% (n=4,897) | 29.81% (n=208) | 68.68pp |
| 3-4 | 99.76% (n=19,567) | 61.11% (n=144) | 38.65pp |
| 5-8 | 99.05% (n=17,891) | 58.72% (n=109) | 40.33pp |
| 9-16 | 96.16% (n=5,362) | 82.05% (n=39 ⚠small-n) | 14.11pp |

At the sharpest single point — patch length exactly 1 byte — natural 96.98% (n=995) vs. boundary
8.33% (n=96 ⚠small-n), non-overlapping 95% CIs (boundary 4.28-15.59% vs. natural's CI floor of
95.73%).

**Verdict.** Length-matched, the gap doesn't shrink — it's essentially unchanged from the raw
comparison, and sharpest at the shortest length. This rules out length as a confound: a 1-byte
patch is not intrinsically hard to pool; a *forced* 1-byte patch is. Refines Hypothesis B (§6.2)
from "short patches are hard" to "forced, non-entropy-triggered closures are hard, independent of
length" — and rules out both remaining variants of §7.2 (length exposure, offset exposure) as a
fix for this specific effect, since neither changes *why* a patch was closed, only *when*.

### 7.7 Synthesis: What §7 Established

Combined, §7.1-§7.6 establish that the post-Root-Cause-#1 gap has (at least) two components: a
small, real, supervision-rate-sensitive one (§7.1's control-adjusted +3.8pp) and a larger,
closure-type-driven one confined to forced-boundary final rows (§7.4, §7.6, up to ~69pp at the
sharpest length-matched point). Both are real. Neither, even combined, is large enough in scope to
be the dominant explanation for why *generation* collapses as fast as it does — forced-boundary
final rows are a small fraction of all rows in any given window or generation run. Investigating
why led to §8 and §9.

## 8. Generation-Time Investigation

The training-side diagnostics above all measure training-window artifacts. None of them establish
whether generation actually encounters anything like a forced-boundary patch, or whether the
symptom in §1 has some other cause entirely. This section traces that question end to end.

### 8.1 Does Generation Replicate Window-Edge Truncation?

**Finding: no fixed rolling buffer exists anywhere in inference; the mechanism that produces
training's boundary-final rows has no direct counterpart during generation.**

All three generation paths (plain greedy `generate_greedy.c:39-76`; BLT-D/BLT-DV
`block_generation.c:331-342`; BLT-S `self_speculation.c:353-374`) re-run the entropy LM over the
**full** committed history from scratch every round — no sliding window, no KV cache for the
entropy model, no carryover between rounds (`entropy_lm.c:58-105`, RoPE rebuilt each call). The
only hard length constraint is a 512-byte cap on total generated output
(`infer.c:386,396,405`; `block_generation.c:311-316`) — the same length as a training window, but
as one continuously-growing document, not a sliding 512-byte buffer.

The training-time `CLOSURE_BND` mechanism (permanently force-closing a patch because the window
ran out of bytes) has no equivalent mid-generation. The "frontier" patch (the currently-open,
not-yet-closed patch at the growing edge) is force-closed each round purely so it has a latent to
predict from, but this closure is transient and self-correcting: next round, the same
prefix-stable, causal segmentation logic re-derives it as part of a longer patch if no entropy
trigger has fired (`self_speculation.c:3-4`). The only *permanent* mid-sequence artificial split
found is BLT-DV's verify-forced-boundary (`self_speculation.c:107-132`), a verification-
correctness device paired with a full model re-run, explicitly not a cold-start penalty like
training's window edge; the BLT-S aligned variant skips it entirely.

**What this ruled out, precisely:** a decode step never permanently truncates a patch mid-stream
the way a training window's edge does. **What it did not rule out**, and what turned out to
matter: the frontier patch is *re-closed every round regardless*, for cross-attention purposes,
and — per §9 — this repo's cross-attention convention gives every row, including the frontier's,
full unconditional access to its own patch's latent. A frontier patch's own latent is necessarily
*incomplete* relative to what the same patch's latent looks like once training sees it as a whole,
because generation cannot see bytes that haven't been produced yet. This is a different, and
ultimately larger, mechanism than the one this section was originally checking for. See §8.3, §9.

### 8.2 Entropy-Lookahead Feasibility (Explored, Not Pursued as a Fix)

While investigating whether the patcher could distinguish a genuinely-triggered close from a
force-close for lack of data, it was confirmed that the entropy value needed for this — `H(x_l)`,
the predictive entropy for the position immediately after the last committed byte — is produced
for free by the entropy LM's existing forward pass, at `entropy_vals[cur_len-1]`
(`entropy_lm.c:58-105,87-89,97-104`), requiring no extra computation and no lookahead into
unseen bytes. It is, however, **not currently used for this decision anywhere**: the patcher
force-closes the frontier unconditionally (`patcher.c:87-90`), and the available entropy value at
index `cur_len-1` only feeds the *previous* patch's boundary decision, not the frontier's own
close/keep-open decision, under the patcher's current indexing convention
(`patcher.c:53-93`,`patcher.h:32-33`). Using it for a genuine-close decision would require a
patcher semantics/API change (the function currently sizes strictly 1:1 with the byte buffer, with
no room for an out-of-coverage lookahead value), not just reading an existing number differently.
Not pursued further, since §9 identifies a more fundamental issue this wouldn't fix on its own.

### 8.3 Frontier-Length Accuracy Under Teacher Forcing, and Reconciliation

**Setup.** A new tool (`greedy_tf_acc.c`) drives the actual greedy generation code path
(full-history re-segmentation each round) under teacher forcing — the true next byte is fed back
each round rather than the model's own output, isolating architecture-level accuracy from
compounding generation drift. 300 windows × 512 positions from heldout data, `p7` checkpoint,
153,000 total predictions.

**Result — accuracy bucketed by the frontier patch's accumulated length at prediction time:**

| Frontier length | n | Accuracy | 95% CI |
|---|---|---|---|
| 1-2 | 57,405 | 14.66% | 14.37-14.95 |
| 3-4 | 48,034 | 49.03% | 48.58-49.48 |
| 5-8 | 35,813 | 58.15% | 57.64-58.66 |
| 9-16 | 11,748 | 69.77% | 68.93-70.59 |

Overall: 60,987/153,000 = 39.86%. Monotonic: the more of the current patch has accumulated, the
better the (still honest, non-leaky — see below) prediction.

**Reconciling this against training-time "interior accuracy."** An initial attempt to bucket these
same rows by the generation tool's `is_final` label produced an apparent collapse relative to
batch's ~98% natural-final figure. This was traced to an off-by-one in the label itself: the tool's
`is_final(p)` marked byte `p` as final if it's the last byte of the *eventual, fully-formed* patch
`[s, p+1)`, but the prediction at round `p` is made from row `p-1`, whose frontier at that moment is
the shorter `[s, p)`. Correcting the alignment (batch row `i` = last byte of `[s, i+1)`, predicting
`text[i+1]` outside the patch ↔ generation round `p = i+1`, where frontier length equals the true,
complete patch length) gives an equivalence class covering only 18.88% of all rows (28,893/153,000).
On exactly this aligned set, generation matches batch closely — natural-equivalent 98.73% vs.
batch's natural-final 98.96%, with **zero** prediction-level disagreements out of 28,893 rows given
identical inputs. Disagreement on the remaining 81.12% of rows (90,057/124,107 = 72.6% locally,
90,057/153,000 = 58.86% overall) is concentrated exactly where training's own-patch bidirectional
pooling would have let those rows see their own target byte and generation's incremental,
still-forming frontier could not.

**How to read this.** The corrected alignment shows there is no measurement bug left once
frontier length is matched to true patch length — the model performs about as well as batch
predicts at that one matched point, and the `is_final` mislabeling that made it look worse than
that was real and is now fixed. But this does *not* mean "no real deficit exists" — it identifies
what the deficit actually is: **for the ~81% of positions where a patch is still forming, honest
frontier-conditioned prediction (Table above) is far below what training-time "interior accuracy"
reported, because training-time interior accuracy was never conditioned on an incomplete patch in
the first place.** The next section explains why not.

## 9. Root Cause #2 (Candidate): Decoder Cross-Attention Convention Deviates from the Paper

### 9.1 What the papers specify

Fast BLT §3.1.1 (attention patterns for BLT-D's decoder, explicitly marked "consistent with BLT,"
i.e. this describes standard BLT decoder behavior, not something BLT-D-specific):

> "for clean positions in the sequence (i ≤ N), each position attends to the latent token
> $o_{p(i)-1}$ corresponding to the previous patch, except for the final byte of each patch, which
> attends to its own latent token $o_{p(i)}$"

Two different rules for two different cases: a patch's non-final bytes attend the **previous**,
already-fully-formed patch's latent; only the **final** byte of a patch attends its own,
just-completed latent.

### 9.2 What this implementation does

One function decides this for every cross-attention call in the codebase, training and inference
alike: `blt_patch_build_group_ids` (`csrc/ops/patch_pool_cpu.c:98-112`). The decisive line:

```c
for (size_t j = 0; j < num_patches; j++) {
    ...
    for (size_t i = 0; i < patches[j].length; i++) kv_group_ids_out[pos++] = j;
}
```

Every byte gets the index of the patch that contains it — its own patch — unconditionally. There
is no final/non-final branch anywhere in this function, and the attention mask
(`csrc/ops/mask_builder_cpu.c:82-88`) enforces strict group-id equality with
`bidirectional_within_group = true` and `is_causal = false`: a decoder row attends the *entire*
pooled latent of its own patch, every sub-token, bidirectionally, with no other restriction.

This is applied identically:

- **Training** (`local_decoder.c:82-87`, plain path `d0_opts=NULL` → `num_hfinal_rows=seq_len`,
  confirmed at line 246): every row `i` gets group `= p(i)`, its own patch, via
  `patch_pool_cpu.c:109`.
- **Plain greedy generation** (`generate_greedy.c:74-84`): segments, then a single full forward via
  `blt_model_forward` → `blt_model_decode` with `d0_opts=NULL`, the *same* code path as training,
  applied to the entire prefix including the frontier. No frontier-specific handling exists for
  clean rows.
- **BLT-D/BLT-DV clean rows** (`block_diffusion.c:164-179`): own-patch rule, explicitly labeled
  in-code as the "repo BLT rule," contrasted in the same comment against the "paper $o_{i-1}$
  rule" — which is reserved for diffusion block/draft rows only (`block_generation.c:205`: all
  draft rows attend the frontier latent collectively, `o_{num\_patches-1}`).
- **BLT-S incremental decode** (`kv_cache.c:192-212,239-243`): re-derives the identical rule —
  committed rows inside a patch attend their own patch; only rows past the last patch's end
  (draft rows) fall back to the frontier.

The codebase names this choice explicitly and treats it as deliberate, not an oversight confined to
one code path: clean rows use "the repo BLT rule" everywhere, both at training time and at
generation time, in all three generation paths. The one place the *paper's* rule is used at all is
for BLT-D's diffusion block rows specifically — and even there, training and inference disagree
with each other (training: block row for patch `i` attends `o_{i-1}`, the true previous patch,
`block_diffusion.c:93`; inference: all block rows attend the frontier `o_{num_patches-1}`,
`block_generation.c:205`, documented in-code as intentional at `block_diffusion.c:751-753`). That
train/inference mismatch is a separate, narrower issue confined to block rows; the clean-row
own-patch-always convention discussed here is consistent between training and generation — it's
consistently the wrong rule, not an inconsistently-applied one.

### 9.3 Why this produces the pattern seen throughout §7 and §8

The encoder pools each patch's latent **bidirectionally** over the patch's entire span, with no
masking of later bytes (`patch_pool_cpu.c:28-35`). Combine that with the decoder's own-patch-always
rule: a non-final row `i`, predicting `text[i+1]` which lies **inside its own patch**, attends a
representation that was computed *from* `text[i+1]` (and any later bytes of the same patch) in the
first place. The target is visible in the representation the model is reading from.

This has no effect on final-byte rows under either convention — a patch's final byte predicts a
byte *outside* the patch, which was never part of what got pooled, so §7.3's/§7.4's/§7.6's findings
about final rows stand exactly as reported. It has a large effect on everything else. Per §8.3's
Check A, only 18.88% of all rows measured anywhere in training-style evaluation are final-byte
rows; the other ~81% — every "interior" row in every table in §7.3, §7.4, §7.6, and the original
symptom-isolation table in §2 — is a non-final row, and is measuring a task with the answer
available in the input, to a degree this document has not yet quantified honestly (see §9.4).

This also explains why the paper's rule is *previous*-patch rather than some other alternative:
the previous patch is guaranteed complete in both training and generation — nothing later can
change it. A row's own patch is only guaranteed complete in training, where the whole window is
processed in one shot. At generation time, a row's own (still-forming) patch is complete only at
its own final byte; every other position within it is necessarily reading an incomplete
representation, because the bytes that would complete it haven't been generated yet. Training under
the repo's convention teaches the model a task — "read the answer out of your own patch's finished
representation" — that is structurally unavailable during autoregressive generation for the large
majority of positions. §8.3's Table A (14.66% at frontier length 1-2, rising to 69.77% by length
9-16) is exactly the shape this predicts: accuracy tracking how much of the "own patch" has managed
to accumulate, because that's the only way the incomplete-at-generation-time representation can
approach what training always had available.

### 9.4 Status: mechanism confirmed, magnitude not yet measured

Everything in §9.1-§9.3 is established by direct comparison of quoted paper text against cited
code, plus the empirical confirmation already in hand from §8.3 (Table A's shape, and Check A's
exact match at the one point — full-length frontier — where the repo's convention and the paper's
convention coincide). What is **not** yet known is how much genuine, non-leaky causal-prediction
skill the existing `p7` checkpoint has, independent of this convention — i.e., whether 900k steps
of training under the leaky convention produced encoder/global representations with real
transferable value that a corrected decoder could exploit, or whether the deficit runs deeper.

The pending test (not yet run as of this revision): patch the decoder's cross-attention group-id
construction at **evaluation time only** — no training change — to implement the paper's rule
(non-final row in patch `j` → group `j-1`, previous patch; final row → group `j`, own patch, same
as today) and re-measure the existing `p7` checkpoint's non-final-row accuracy on held-out data, in
one batched forward pass. If corrected accuracy collapses toward the neighborhood of §8.3's Table A
(14-70%, roughly rising with some yet-to-be-determined variable), that's confirmation the deficit is
fundamental to what this checkpoint learned, and argues for a retrain under the corrected
convention. If it holds up meaningfully higher, that argues the representations are more salvageable
than the leaky training task suggests, and a targeted fine-tune under the corrected convention (much
cheaper than a full retrain) may suffice.

## 10. Current Status and Next Steps

| Item | Status |
|---|---|
| Symptom: healthy teacher-forced eval, garbage greedy generation | Confirmed (§1) |
| Root cause #1: off-by-one excluding the last row from `L_clean` | Confirmed and fixed (§3, §5) |
| Fix #1 generalizes across window lengths; interior accuracy (as then measured) unaffected | Confirmed (§5) |
| Full recovery of last-row accuracy after fix #1 | Not achieved on its own (§5-§7) |
| Hypothesis A (per-window supervision imbalance) | Tested (§7.1): control-adjusted +3.8pp, real but minor, costly to apply; distribution analysis (§7.5) argues against exposure as the driver |
| Hypothesis B (truncated/forced-closure patches → weak latent) | Confirmed and refined (§7.4, §7.6): closure type, not length, drives a real ~15-69pp effect confined to forced-boundary final rows |
| Root Cause #2 candidate (decoder cross-attention convention deviates from both papers) | **Mechanism confirmed via direct code-vs-paper comparison (§9); magnitude on the existing checkpoint not yet measured** |
| Generation-time window-edge truncation (literal `CLOSURE_BND` analog) | Does not occur mid-generation (§8.1); superseded as the leading generation-time explanation by §9's broader mechanism |
| Entropy-lookahead genuine-vs-forced-close detection | Feasible in principle (free, no lookahead needed), not implemented, not pursued further given §9 (§8.2) |
| Frontier-length accuracy under teacher forcing (honest, non-leaky measurement) | Measured (§8.3): 14.66%→69.77% by frontier completeness, 39.86% overall |
| `max_patch_length`: project uses 16, Fast BLT paper uses 8 | Deviation noted; upside bounded at ~0.2pp interior accuracy (§7.3), unrelated to §9 |

**Immediate next step:** the eval-time decoder-convention patch described in §9.4 — the single
highest-value pending test, since it determines whether the path forward is a retrain, a targeted
fine-tune, or something else, and nothing else on this list changes that decision.

**After that result is in:** if it confirms a deep deficit, the §7.2 sliding-window idea (§7.2)
should likely be reconsidered as part of a *corrected-convention* retrain rather than evaluated on
its own — its rationale was always weaker in isolation than as a component of getting the training
task to actually match the generation task.

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
