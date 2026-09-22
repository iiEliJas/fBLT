# Patch-final row accuracy (500 windows, tinystories_p7 control)

Source: `runs/tinystories_p7/analyses/patch_final_500w_control.tsv` — 256000 rows, 500 windows

## Patch-final rows (final byte of patch)

| Length | nat | max | bnd |
| --- | --- | --- | --- |
| 1-2 | n=4897 acc=98.86% (98.52%–99.12%) | (no samples) ⚠small-n | n=208 acc=31.25% (25.34%–37.84%) |
| 3-4 | n=19567 acc=99.79% (99.72%–99.85%) | (no samples) ⚠small-n | n=144 acc=66.67% (58.62%–73.84%) |
| 5-8 | n=17891 acc=99.09% (98.94%–99.22%) | (no samples) ⚠small-n | n=109 acc=57.80% (48.42%–66.65%) |
| 9-16 | n=5362 acc=96.74% (96.23%–97.18%) | n=491 acc=86.56% (83.26%–89.29%) | n=39 acc=87.18% (73.29%–94.40%) ⚠small-n |
| 17+ | (no samples) ⚠small-n | (no samples) ⚠small-n | (no samples) ⚠small-n |

## Non-final rows

| Length | nat | max | bnd |
| --- | --- | --- | --- |
| 1-2 | n=3902 acc=99.87% (99.70%–99.95%) | (no samples) ⚠small-n | n=112 acc=98.21% (93.72%–99.51%) |
| 3-4 | n=51603 acc=99.92% (99.89%–99.94%) | (no samples) ⚠small-n | n=351 acc=98.58% (96.71%–99.39%) |
| 5-8 | n=89852 acc=99.22% (99.16%–99.27%) | (no samples) ⚠small-n | n=551 acc=97.10% (95.34%–98.20%) |
| 9-16 | n=53189 acc=95.97% (95.80%–96.13%) | n=7365 acc=92.07% (91.43%–92.67%) | n=367 acc=95.64% (93.04%–97.30%) |
| 17+ | (no samples) ⚠small-n | (no samples) ⚠small-n | (no samples) ⚠small-n |

## Summary

nat: final n=47717 acc=99.09% (99.00%–99.17%) | nonfinal n=198546 acc=98.54% (98.49%–98.59%) | gap=-0.55pp
max: final n=491 acc=86.56% (83.26%–89.29%) | nonfinal n=7365 acc=92.07% (91.43%–92.67%) | gap=5.51pp ⚠gap
bnd: final n=500 acc=51.60% (47.22%–55.95%) | nonfinal n=1381 acc=97.18% (96.16%–97.93%) | gap=45.58pp ⚠gap

## Small-n caveats

- none (all cells n>=100)
