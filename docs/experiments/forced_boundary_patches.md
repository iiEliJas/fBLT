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

Four points now have a measured boundary-final accuracy, spanning four different training histories:

| Checkpoint | Training history | bnd-final accuracy | 95% CI | n |
|---|---|---|---|---|
| Original (leaky decoder convention) | 900k steps | 49.20% | 44.8-53.6% | 500 |
| Continuation, post cross-attention fix | 900k leaky steps + 30k continuation under the corrected convention | 56.20% | 51.8-61.5% | 500 |
| From-scratch, both fixes present | 300k of a planned 600k steps | 42.80% | 38.5-47.2% | 500 |
| Same run, completed | +300k continuation, 600k total | 50.00% | 45.6-54.4% | 500 |

All four are measured on the same 500-window sample, so these are paired comparisons.
The from-scratch run gained 7.20pp between its own two measurements, 42.80% to 50.00%.
At 50.00% (45.6-54.4%) the from-scratch run is no longer separated from the original 900k leaky checkpoint at 49.20%
(44.8-53.6%), and its interval no longer sits clearly below the 56.20% continuation either. 

What is left is broader: the effect appears in all four checkpoints, across leaky, corrected,
half-trained, and fully-trained histories, at 42.80% to 56.20%. It does not need any particular
convention or schedule to show up.
The boundary-final number also hides the interesting part. Comparing final-row against
non-final-row accuracy by closure type on the same 500-window run (256,000 rows):

| Closure | Final-row acc | Non-final-row acc | Gap (final - nonfinal) | Final-row delta vs 300k |
|---|---|---|---|---|
| natural | 99.38% (47,419/47,717) | 69.53% (138,047/198,546) | +29.85pp | +0.17pp |
| max-length | 88.39% (434/491) | 73.84% (5,438/7,365) | +14.56pp | +2.44pp |
| boundary | 50.00% (250/500) | 61.40% (848/1,381) | **-11.40pp** | +7.20pp |

Natural and max-length closures show the expected sign: a row reading its own, legitimately-closed
patch beats one reading a generic previous patch. Boundary closures invert it, and it isn't noise.
The inversion is smaller than at 300k (-14.62pp) but has not closed.

Every closure type improved on the final row, and boundary improved most. The overall final-row
rate barely moved, 98.50% to 98.76%, because natural patches already pin it near ceiling.
Non-final-row went 66.41% to 69.63%. Only the closure split shows any of it.

Boundary-final gained 7.20pp against 3.22pp for interior rows overall, so the boundary position
improved about twice as fast as the model average. At n=500 the interval is 45.6-54.4%, wide enough
that the ratio is suggestive, not established.

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
| 1-2 | 95.61% (4,881/5,105) | 46.34% (1,860/4,014) | +49.27pp |
| 3-4 | 99.52% (19,617/19,711) | 71.39% (37,089/51,954) | +28.13pp |
| 5-8 | 99.03% (17,826/18,000) | 67.94% (61,422/90,403) | +31.09pp |
| 9-16 | 98.08% (5,779/5,892) | 72.16% (43,962/60,921) | +25.92pp |

Every denominator here matches the 300k run, so the segmentation did not move between the two
checkpoints and the accuracy differences are purely a model change.

Short patches are the weakest final-row bucket at 95.61%, but that's nowhere near 50.00%, so
length isn't what's driving the boundary number. These buckets pool all closure types, which means
97.97% of the final rows they contain are natural closures (47,717/48,708); the 500 boundary rows
are 1.03% of the total and barely move the aggregate. Only the closure split above isolates them.

Still not run: the same length breakdown restricted to boundary-closed patches, which is where
the 50.00% actually lives. 

For the completed run, `sanity_check` over 50 windows and 25,550 held-out bytes:

| Metric | Value |
|---|---|
| Mean CE | 0.7952 nats/byte |
| Mean BPB | 1.1473 bits/byte |
| Top-1 | 75.04% |
| Top-5 | 93.20% |
| Top-10 | 97.42% |

## Why neither paper mentions it

Both papers report aggregate metrics (BPB, pass@1, BLEU) over training runs at a much larger
scale, trillions of bytes. A weakness confined to one relative position per training window,
diluted across a dataset that size, wouldn't move an aggregate enough to notice, even if it's
present in their own runs.

## Status

Not fixed, and there's no code-level deviation to point at, since neither paper specifies how to
handle a window boundary landing mid-patch. It's a byproduct of chopping training data into
fixed-length windows for causal training, not a bug in the usual sense.

The planned 600k re-measurement is done. The effect is not a training-duration artifact: the
from-scratch run at 600k reaches 50.00% on boundary-final rows, roughly 49pp below its own natural
closures at 99.38%, and still the only closure type whose final-row versus non-final-row gap is
negative. Doubling the step budget moved the number 7.20pp and did not close the deficit.

The earlier argument that the cleanest configuration produced the lowest number does not survive,
since 50.00% is level with the 900k leaky checkpoint. What holds instead is that the effect appears
in every history measured. The closure split still holds at 500 final and 1,381 non-final boundary
rows.

Still open: the length breakdown restricted to boundary-closed patches is unrun, and the 7.20pp
stage-2 gain is confounded by the simultaneous lr and mask_scale changes, so attributing it to
duration needs a controlled run at a fixed schedule.

A fix worth considering later: overlapping or sliding training windows, so a given stretch of text
isn't always cut at the same relative offset. Content that's forced-closed in one window would
then show up naturally-closed or interior in another, diluting the forced-cut supervision for the
same content rather than reshuffling which lengths get cut. But that's only an idea, and neither
implemented nor tested.
