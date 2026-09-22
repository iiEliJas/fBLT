# window offset sourcing & length-matched nat-vs-bnd final-row accuracy

Sources:
- `runs/tinystories_p7/analyses/patch_final_500w.tsv` - 256000 rows, 500 heldout windows, `tinystories_p7.fblt` (baseline)
- Training-window truncation measurement: `patch_final_split` over `data/tinystories/train.bin`, 500 windows (skips 0/125/250/375), entropy-LM patching - same segmentation code path as training (`train_eval.c:135-167`)

## 1. Window offsets are fixed and deterministic - no randomization exists

`csrc/train_blt_d.c`:
- **Line 298**: `num_windows = (fsize - a.window) / a.window` - non-overlapping tiling of the corpus into fixed 512-byte cells
- **Line 331**: step loop
- **Line 338**: `const size_t w = step % num_windows;` - sequential cycling through cells
- **Line 339**: `text = corpus + w * a.window;` - window always starts at byte offset `w*512`

`csrc/train_args.c`: no `shuffle`/`offset`/`jitter`/`stride`-randomization flags exist. The RNG is used only for the corruption mask and timestep draws (`train_blt_d.c:352`), never for window placement. `entropy_segment` (`train_eval.c:135-167`) runs the entropy LM cold over exactly the 512 window bytes with no carryover from earlier corpus positions; the patcher closes the final patch unconditionally at the window edge (`patcher.c:87-90`).

**Conclusion: window start offsets are FIXED. Every epoch revisits the same cells; segmentation per cell is deterministic (only the corruption mask re-draws).**

## 2. Truncation length at the cut is already varied (training ≈ heldout)

Measured over 500 `train.bin` windows (same entropy LM, same segmentation path as training):

| bnd length | train windows | heldout windows |
|---|---|---|
| 1-2 | 206 (41.2%) | 208 (41.6%) |
| 3-4 | 143 (28.6%) | 144 (28.8%) |
| 5-8 | 108 (21.6%) | 109 (21.8%) |
| 9-16 | 43 (8.6%) | 39 (7.8%) |

Mean truncation length: **3.90** (train) vs **3.76** (heldout). The distributions are statistically indistinguishable - same slicing rule, same entropy LM, same text statistics.

**Conclusion: the truncation-length-at-the-cut distribution is NOT fixed/narrow. It spans 1-15 with a short skew, and it matches what the model is evaluated on. Randomizing offsets would not change this aggregate distribution (it is content-driven).**

## 3. Length-matched natural vs boundary final-row accuracy (new analysis)

Re-analysis of the existing baseline dump. Buckets identical to the boundary table; all CIs are 95% Wilson, matching `analyze_patch_final.py`.

| Bucket | nat final | bnd final | gap (nat − bnd) |
|---|---|---|---|
| 1-2 | n=4897 acc=98.49% (98.11%–98.79%) | n=208 acc=29.81% (24.00%–36.34%) | +68.68pp |
| 3-4 | n=19567 acc=99.76% (99.69%–99.82%) | n=144 acc=61.11% (52.96%–68.69%) | +38.65pp |
| 5-8 | n=17891 acc=99.05% (98.90%–99.18%) | n=109 acc=58.72% (49.33%–67.51%) | +40.33pp |
| 9-16 | n=5362 acc=96.16% (95.61%–96.64%) | n=39 acc=82.05% (67.33%–91.02%) ⚠small-n | +14.11pp |
| 17+ | (no samples) ⚠small-n | (no samples) ⚠small-n | - |

**Bucket-for-bucket, natural patches at the same lengths are 14-69pp better than boundary patches.** The gap shrinks monotonically with length but never closes: even at 9-16 bytes (where the boundary patch carries the most in-window context, n=39 ⚠small-n), the boundary-final row remains 14pp behind the natural-final row.

### Exact short lengths (power check)

| Length | nat final | bnd final | gap (nat − bnd) |
|---|---|---|---|
| 1 | n=995 acc=96.98% (95.73%–97.88%) | n=96 acc=8.33% (4.28%–15.59%) ⚠small-n | +88.65pp |
| 2 | n=3902 acc=98.87% (98.49%–99.16%) | n=112 acc=48.21% (39.17%–57.37%) | +50.66pp |
| 3 | n=7098 acc=99.69% (99.53%–99.80%) | n=81 acc=54.32% (43.52%–64.73%) ⚠small-n | +45.37pp |
| 4 | n=12469 acc=99.81% (99.71%–99.87%) | n=63 acc=69.84% (57.64%–79.76%) ⚠small-n | +29.97pp |

## 4. Power assessment for the short-length comparison

**The comparison is NOT underpowered on the natural side - it is the well-powered side.** 1-2-byte natural patches are abundant: n=4897 final rows (995 at length 1, 3902 at length 2), so the 1-2 nat estimate has a CI width of ~0.7pp. Every natural cell has n>=100 and needs no small-n flag.

The small-n limitation sits on the **boundary** side of the table: bnd 9-16 n=39 ⚠small-n (existing flag) and, at exact lengths, bnd len-1 n=96, len-3 n=81, len-4 n=63 ⚠small-n. The len-1 boundary cell (8.33%) is the extreme outlier but rests on only 96 rows - its CI (4.28%–15.59%) is still far below the nat len-1 CI floor (95.73%), so the qualitative gap at every matched length is robust even if the exact len-1 point estimate is shaky.

## 5. Exposure-inversion observation

The boundary-final accuracy data, combined with the truncation-length distribution, shows an inversion:

- The model gets the **most** training exposure on short cuts (1-2 bytes = 41% of windows) - and 1-2-byte boundary finals are its **worst** bucket (29.81% baseline; 31.25% control).
- Natural patches at the **same** 1-2-byte lengths are 98.49% - the short length itself is not hard.

So the difficulty is specific to the **boundary-final task** (predicting the byte right after the artificial cut), not to short patches in general. The model trains most on exactly the hardest configuration and still performs worst there - while the identical length bin for natural closures is essentially solved. This isolates the failure to the window-edge cut, not the length distribution.

## 6. Implication for the random-offset hypothesis

Randomizing start offsets would change only the alignment/the fixed 512-byte grid (new cold-start contexts, different segmentations per step). The data says:

1. The truncation-length distribution at the cut is already varied and already matches heldout - no distribution to "open up".
2. Boundary rows underperform natural rows at **every matched length** (14-69pp gap, monotone in length) - the deficit is closure-type-specific, not length-bin-specific.
3. The model is worst exactly where exposed most, and the same-length natural comparison is not the discriminator - it is already solved.

A random-offset intervention cannot address (2): it does not convert a boundary-final row into a natural-final row. Any ft still produces a final-row-before-the-cut, and the length-matched gap persists by construction. The evidence points the explanation at the boundary-final *task* (supervision/signal at the cut row), consistent with the existing LAST_ROW_TRAINING_GAP hypothesis space - the ×511 upweight (which raises last-row supervision) does improve boundary finals in every bucket (+3.80pp vs control overall; +4.33pp at 1-2).