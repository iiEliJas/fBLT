# Patch-final row accuracy: 3-way comparison (baseline / ×511 upweight / control)

500 heldout windows (256 000 rows), tinystories_p7 split, entropy-LM patching, max patch length 16.
Identical patcher segmentation across all three columns (same n per cell), so only model predictions differ.

- **baseline** — `runs/tinystories_p7/tinystories_p7.fblt` (30 000-step original)
- **×511** — `runs/tinystories_p7/tinystories_p7_lrs511.fblt` (20 000-step continuation, `--last-row-scale 511`)
- **control** — `runs/tinystories_p7/tinystories_p7_control.fblt` (20 000-step continuation, `--last-row-scale 1.0`, upweight off)

All three continuations at 20k steps share seed 7, sgd lr 0.05, `--lr-decay 1` (×0.3 at 12 000, ×0.09 at 17 000), and identical data order/corruption draws. The control was matched flag-for-flag against the ×511 run (same corpus, mask schedule, eval) — `--lr-decay-factor 0.3` is inert in this mode (only read by `--lr-decay-steps`).

## Boundary-final accuracy (the goal metric), bucketed by patch length

| Bucket | n | baseline | ×511 | control | ×511 − baseline | ×511 − control |
|---|---|---|---|---|---|---|
| 1-2 | 208 | 29.81% | 35.58% | 31.25% | +5.77pp | +4.33pp |
| 3-4 | 144 | 61.11% | 67.36% | 66.67% | +6.25pp | +0.69pp |
| 5-8 | 109 | 58.72% | 64.22% | 57.80% | +5.50pp | +6.42pp |
| 9-16 | 39 ⚠small-n | 82.05% | 92.31% | 87.18% | +10.26pp | +5.13pp |
| **Overall bnd final** | **500** | **49.20%** | **55.40%** | **51.60%** | **+6.20pp** | **+3.80pp** |

The control shows ~2.4pp of the +6.20pp came from continued training alone; the upweight contributes the remaining +3.80pp, and helps in every bucket (largest in 5-8 and 9-16).

## Natural & max closures (collateral check)

| Closure | rows | baseline | ×511 | control | ×511 − control |
|---|---|---|---|---|---|
| nat final | 47 717 | 98.96% | 97.25% | 99.09% | −1.84pp |
| nat nonfinal | 198 546 | 98.54% | 95.50% | 98.54% | −3.04pp |
| max final | 491 | 84.32% | 83.50% | 86.56% | −3.06pp |
| max nonfinal | 7 365 | 91.38% | 81.82% | 92.07% | −10.25pp |
| bnd nonfinal | 1 381 | 97.03% | 91.53% | 97.18% | −5.65pp |

**Key attribution result:** the control does NOT reproduce any of the ×511 nat/max degradation — control nat nonfinal (98.54%) and max nonfinal (92.07%) match or slightly beat baseline, and even nat final (99.09%) improves slightly. So the collateral damage in the ×511 run is caused by the upweighting itself, not by continued training:

- max nonfinal −10.25pp is the single biggest casualty (×511 only)
- bnd nonfinal −5.65pp (×511 only; control +0.15pp vs baseline)
- nat nonfinal −3.04pp (×511 only; control exactly matches baseline)

where every "×511 only" is measured against the matched control at the same 20k steps → training-drift is ruled out.

## Boundary gap (final − nonfinal)

| | baseline | ×511 | control |
|---|---|---|---|
| bnd final | 49.20% | 55.40% | 51.60% |
| bnd nonfinal | 97.03% | 91.53% | 97.18% |
| gap | 47.83pp | 36.13pp | 45.58pp |

The upweight shrinks the gap by gaining on final (+3.80pp vs control) while paying for it on nonfinal (−5.65pp vs control). Net gap reduction ≈ 9.4pp, split roughly half upweight-half damage.

## Total panel accuracy

| | correct / 256 000 | panel acc |
|---|---|---|
| baseline | 251 590 | 98.28% |
| ×511 | 243 990 | 95.31% |
| control | 251 742 | 98.34% |

Net panel damage of the upweight: **−3.03pp** vs the matched control (+0.06pp for continued training alone — training 20k more steps is, at worst, neutral, and the control slightly *improves* nat final).

## Verdict

- The ×511 upweight achieves its goal: +3.80pp boundary-final accuracy over the control across all buckets — but it is not free.
- It trades nonfinal and max-closure accuracy for that gain, and the effect is a net panel loss (−3.03pp). The 4.6:1 row magnitude (511×) is likely over-aggressive: it wins its target row but corrupts shared encoder/global representations used by interior rows (max nonfinal suffers most, −10.25pp).
- A smaller scale (e.g. 8-32×, closer to the 1/512 supervision imbalance) might capture most of the bucket-1 gain while keeping the collateral under control — or the upweight could be confined to gradient rows without a full shared-representation penalty. Testing one intermediate scale against this same control protocol would settle whether the trade is fundamentally lossy or just over-amped.