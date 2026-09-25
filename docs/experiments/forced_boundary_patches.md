# Forced-Boundary Patch Closures Weaken Final-Byte Prediction

**Status: OPEN.** Not a deviation from either paper, since neither specifies training-window
boundary handling, so it's outside `last_row_gap.md`'s conformance closure. Recorded here as a
smaller finding that surfaced during that investigation.

## Origin

Same row `last_row_gap.md` §2.1 already fixed: row `N-1`, the last row of every training window.
That fix solved a different problem in the same place. An off-by-one loop bound excluded the row
from the loss entirely, so it got no gradient at all. Once fixed, it trains like every other row,
and its final-byte prediction is still measurably weaker than the rest of the window. That's what
this document is about.

## The finding

Row `N-1` is by construction always the final byte of whichever patch happens to be open when the
window buffer ends, whether or not the entropy patcher would have closed it there on its own. A
patch closed that way pools whatever bytes happened to accumulate before the cutoff, not a span
the entropy model found coherent. That's the weaker, more arbitrary representation the
final-byte prediction reads from.

Length-matched, natural closures against forced boundary closures at the same patch lengths:

| Patch length | Natural closure | Forced boundary closure | Gap |
|---|---|---|---|
| 1 byte | 96.98% (n=995) | 8.33% (n=96) | 88.65pp |
| 1-2 | 98.49% (n=4,897) | 29.81% (n=208) | 68.68pp |
| 3-4 | 99.76% (n=19,567) | 61.11% (n=144) | 38.65pp |
| 5-8 | 99.05% (n=17,891) | 58.72% (n=109) | 40.33pp |
| 9-16 | 96.16% (n=5,362) | 82.05% (n=39) | 14.11pp |

Length-matched, the gap doesn't shrink. A short natural patch isn't hard to pool from; a short
*forced* patch is. Closure type drives this, not length.

## Tracking this across later checkpoints

Three checkpoints now have a measured boundary-final accuracy, from three different training
histories:

| Checkpoint | Training history | bnd-final accuracy | n |
|---|---|---|---|
| Original (leaky decoder convention) | 900k steps | 49.20% | 500 |
| Continuation, post cross-attention fix | 900k leaky steps + 30k continuation under the corrected convention | 56.20% | 500 |
| From-scratch, both fixes present from step 0 | 300k of a planned 600k steps | 42.80% | 500 |

The from-scratch point is now measured on the same 500-window sample as the other two. At 214/500 its 95% CI is
roughly 38.5-47.2%, so it no longer reaches 49.20% and the three are directly comparable:
42.80% sits below both 900k checkpoints.

That ordering is what you'd expect from a run stopped at half its planned training, and the
checkpoint is still structurally different, built from scratch under both fixes rather than a
continuation on a mature, leaky-trained base. What makes it worth having anyway is which run
produced the lowest number: the one with both fixes in place from step 0 and no leaky history to
lean on. The effect survives in the cleanest configuration measured so far, which is the
strongest evidence yet that it isn't a leftover of the convention bugs.

The boundary-final number also hides the interesting part. Comparing final-row against
non-final-row accuracy by closure type on the same 500-window run (256,000 rows):

| Closure | Final-row acc | Non-final-row acc | Gap (final − nonfinal) |
|---|---|---|---|
| natural | 99.21% (47,339/47,717) | 66.34% (131,709/198,546) | +32.87pp |
| max-length | 85.95% (422/491) | 70.09% (5,162/7,365) | +15.86pp |
| boundary | 42.80% (214/500) | 57.42% (793/1,381) | **−14.62pp** |

Natural and max-length closures show the expected sign: a row reading its own, legitimately-closed
patch beats one reading a generic previous patch. Boundary closures invert it, and it isn't noise.

Under the corrected decoder convention (`last_row_gap.md` §2.2), a final-byte row reads its *own*
patch's latent, a non-final row the *previous* patch's. For natural and max-length patches "own
patch" is a good patch, so it wins. For boundary patches, "own patch" is the one thing about this
position that's arbitrary and bad, which makes it the only closure type where reading your own
patch loses to whatever ordinary patch preceded it.

Independent confirmation this is a patch-quality problem downstream of the cross-attention fix,
not a leftover symptom of the routing bug.

The same run also gives a length breakdown, bucketed by patch length across all 256,000 rows.

| Patch length | Final-row acc | Non-final-row acc | Gap |
|---|---|---|---|
| 1-2 | 94.91% (4,845/5,105) | 44.22% (1,775/4,014) | +50.69pp |
| 3-4 | 99.33% (19,578/19,711) | 68.50% (35,587/51,954) | +30.83pp |
| 5-8 | 98.87% (17,796/18,000) | 64.64% (58,438/90,403) | +34.23pp |
| 9-16 | 97.69% (5,756/5,892) | 68.72% (41,864/60,921) | +28.97pp |

Short patches are the weakest final-row bucket at 94.91%, but that's nowhere near 42.80%, so
length isn't what's driving the boundary number. These buckets pool all closure types, which
means 99.21% of the final rows they contain are natural; the boundary rows are diluted to 500
out of 48,708 and don't move the aggregate. Only the closure split above isolates them.

Still not run: the same length breakdown restricted to boundary-closed patches, which is where
the 42.80% actually lives. Repeat both at 450k and 600k steps as the from-scratch run finishes.

## Why neither paper mentions it

Both papers report aggregate metrics (BPB, pass@1, BLEU) over training runs at a much larger
scale, trillions of bytes. A weakness confined to one relative position per training window,
diluted across a dataset that size, wouldn't move an aggregate enough to notice, even if it's
present in their own runs.

## Status

Not fixed, and there's no code-level deviation to point at, since neither paper specifies how to
handle a window boundary landing mid-patch. It's a byproduct of chopping training data into
fixed-length windows for causal training, not a bug in the usual sense.

Boundary-final accuracy across the three checkpoints spans 42.80% to 56.20% at a matched 500
windows each (§ above). The lowest of the three is the from-scratch run that has both fixes in
place from step 0, and the closure split holds at 1,381 non-final boundary rows. Both point at a
patch-quality effect rather than a convention artifact. The from-scratch number comes from a run
at half its planned steps, so whether it converges up toward the 900k checkpoints is still open.
I will re-measure at 600k.

A fix worth considering later: overlapping or sliding training windows, so a given stretch of text
isn't always cut at the same relative offset. Content that's forced-closed in one window would
then show up naturally-closed or interior in another, diluting the forced-cut supervision for the
same content rather than reshuffling which lengths get cut. But that's only an idea, and neither
implemented nor tested.
