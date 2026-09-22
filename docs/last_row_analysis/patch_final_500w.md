# Patch-final row accuracy (500 windows, tinystories_p7)

Source: `runs/tinystories_p7/analyses/patch_final_500w.tsv` — 256000 rows, 500 windows

## Patch-final rows (final byte of patch)

| Length | nat | max | bnd |
| --- | --- | --- | --- |
| 1-2 | n=4897 acc=98.49% (98.11%–98.79%) | (no samples) small-n | n=208 acc=29.81% (24.00%–36.34%) |
| 3-4 | n=19567 acc=99.76% (99.69%–99.82%) | (no samples) small-n | n=144 acc=61.11% (52.96%–68.69%) |
| 5-8 | n=17891 acc=99.05% (98.90%–99.18%) | (no samples) small-n | n=109 acc=58.72% (49.33%–67.51%) |
| 9-16 | n=5362 acc=96.16% (95.61%–96.64%) | n=491 acc=84.32% (80.84%–87.27%) | n=39 acc=82.05% (67.33%–91.02%) small-n |
| 17+ | (no samples) small-n | (no samples) small-n | (no samples) small-n |

## Non-final rows

| Length | nat | max | bnd |
| --- | --- | --- | --- |
| 1-2 | n=3902 acc=99.77% (99.56%–99.88%) | (no samples) small-n | n=112 acc=100.00% (96.68%–100.00%) |
| 3-4 | n=51603 acc=99.90% (99.87%–99.92%) | (no samples) small-n | n=351 acc=97.44% (95.20%–98.65%) |
| 5-8 | n=89852 acc=99.26% (99.20%–99.31%) | (no samples) small-n | n=551 acc=97.28% (95.56%–98.34%) |
| 9-16 | n=53189 acc=95.90% (95.73%–96.06%) | n=7365 acc=91.38% (90.72%–92.00%) | n=367 acc=95.37% (92.71%–97.09%) |
| 17+ | (no samples) small-n | (no samples) small-n | (no samples) small-n |

## Summary

nat: final n=47717 acc=98.96% (98.87%–99.05%) | nonfinal n=198546 acc=98.54% (98.48%–98.59%) | gap=-0.42pp
max: final n=491 acc=84.32% (80.84%–87.27%) | nonfinal n=7365 acc=91.38% (90.72%–92.00%) | gap=7.06pp gap
bnd: final n=500 acc=49.20% (44.84%–53.57%) | nonfinal n=1381 acc=97.03% (96.00%–97.80%) | gap=47.83pp gap

## Small-n caveats

- none (all cells n>=100)
