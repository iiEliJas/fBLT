# Patch-final row accuracy (500 windows, tinystories_p7 + last-row upweight)

Source: `patch_final_500w_lrs511.tsv` — 256000 rows, 500 windows

## Patch-final rows (final byte of patch)

| Length | nat | max | bnd |
| --- | --- | --- | --- |
| 1-2 | n=4897 acc=94.51% (93.83%–95.11%) | (no samples) ⚠small-n | n=208 acc=35.58% (29.39%–42.29%) |
| 3-4 | n=19567 acc=98.90% (98.74%–99.03%) | (no samples) ⚠small-n | n=144 acc=67.36% (59.34%–74.48%) |
| 5-8 | n=17891 acc=97.02% (96.76%–97.25%) | (no samples) ⚠small-n | n=109 acc=64.22% (54.88%–72.59%) |
| 9-16 | n=5362 acc=94.52% (93.88%–95.10%) | n=491 acc=83.50% (79.96%–86.52%) | n=39 acc=92.31% (79.68%–97.35%) ⚠small-n |
| 17+ | (no samples) ⚠small-n | (no samples) ⚠small-n | (no samples) ⚠small-n |

## Non-final rows

| Length | nat | max | bnd |
| --- | --- | --- | --- |
| 1-2 | n=3902 acc=99.36% (99.06%–99.57%) | (no samples) ⚠small-n | n=112 acc=97.32% (92.42%–99.08%) |
| 3-4 | n=51603 acc=99.49% (99.43%–99.55%) | (no samples) ⚠small-n | n=351 acc=92.88% (89.70%–95.13%) |
| 5-8 | n=89852 acc=96.81% (96.69%–96.92%) | (no samples) ⚠small-n | n=551 acc=92.01% (89.45%–94.00%) |
| 9-16 | n=53189 acc=89.13% (88.86%–89.39%) | n=7365 acc=81.82% (80.92%–82.68%) | n=367 acc=87.74% (83.99%–90.71%) |
| 17+ | (no samples) ⚠small-n | (no samples) ⚠small-n | (no samples) ⚠small-n |

## Summary

nat: final n=47717 acc=97.25% (97.10%–97.39%) | nonfinal n=198546 acc=95.50% (95.41%–95.59%) | gap=-1.75pp
max: final n=491 acc=83.50% (79.96%–86.52%) | nonfinal n=7365 acc=81.82% (80.92%–82.68%) | gap=-1.68pp
bnd: final n=500 acc=55.40% (51.02%–59.70%) | nonfinal n=1381 acc=91.53% (89.94%–92.88%) | gap=36.13pp ⚠gap

## Small-n caveats

- none (all cells n>=100)
