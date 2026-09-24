# Forced-Boundary Patch Closures Weaken Final-Byte Prediction

**Status: OPEN.** Not a deviation from either paper, since neither specifies training-window
boundary handling, so it isn't part of `LAST_ROW_TRAINING_GAP.md`'s conformance closure. Recorded
here as a separate, smaller finding that surfaced during that investigation and remains
unresolved.

## Origin

This finding sits on the exact same row that `LAST_ROW_TRAINING_GAP.md` §2.1 
already fixed: row `N-1`, the last row of every training window. That fix solved a different
problem at the same location: the row got zero gradient at all, an off-by-one loop bound that
excluded it from the loss entirely. Once fixed, the row trains like every other row. What turned
up next, once it had real supervision to work with, is what this document covers: even with a
gradient, that row's prediction stays meaningfully weaker than the rest of the window.

## The finding

Row `N-1` is, by construction, always the final byte of whichever patch happens to be open when
the window buffer ends, whether or not the entropy patcher would have chosen to close it there on
its own. A patch closed this way pools whatever bytes happened to accumulate before the cutoff,
not a span the entropy model determined was actually coherent. That weaker, more arbitrary
representation is what the final-byte prediction reads from.

Length-matched comparison, natural closures against forced boundary closures at the same patch
lengths:

| Patch length | Natural closure | Forced boundary closure | Gap |
|---|---|---|---|
| 1 byte | 96.98% (n=995) | 8.33% (n=96) | 88.65pp |
| 1-2 | 98.49% (n=4,897) | 29.81% (n=208) | 68.68pp |
| 3-4 | 99.76% (n=19,567) | 61.11% (n=144) | 38.65pp |
| 5-8 | 99.05% (n=17,891) | 58.72% (n=109) | 40.33pp |
| 9-16 | 96.16% (n=5,362) | 82.05% (n=39) | 14.11pp |

Length-matched, the gap doesn't shrink. A short natural patch isn't hard to pool from; a short
*forced* patch is. Closure type drives this, not patch length.

## Why neither paper mentions it

Same reason as Root Cause #1: both papers report aggregate metrics (BPB, pass@1, BLEU) over
training runs at a much larger scale, trillions of bytes. A weakness confined to one relative
position per training window, diluted across a dataset that size, wouldn't move an aggregate
number enough to notice, even if it's present in their own training runs.

## Status

Not fixed. There's no code-level deviation from either paper to point at here, since neither
specifies how to handle a window boundary landing mid-patch. It's a byproduct of chopping training
data into fixed-length windows for causal training, not a bug in the usual sense.

One suggestive but unconfirmed data point: forced-boundary final-row accuracy rose from 49.20% to
56.20% alongside the decoder cross-attention retrain (`LAST_ROW_TRAINING_GAP.md` §2.2), on a run
aimed at a different fix. 

A fix worth considering later: overlapping or sliding training windows, so a given stretch of text
isn't always cut at the same relative offset. Content that's forced-closed in one window would
then show up naturally-closed or interior in another, diluting the forced-cut supervision for the
same content rather than reshuffling which lengths get cut. But thats only an idea and neither implemented or tested.
