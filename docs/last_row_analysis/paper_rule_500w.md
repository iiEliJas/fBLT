# Paper-rule decoder cross-attention eval

Source: `runs/tinystories_p7/analyses/paper_rule_500w.tsv` — 256000 rows, 500 windows

## Setup

- Windows evaluated: 500
- Total rows: 256000 (expected 256000: yes)
- Baseline: `runs/tinystories_p7/analyses/patch_final_500w.tsv` (256000 rows)
- Headline set: is_final==0 AND no_prev==0 → 205910 rows (80.43% of all rows)
- First-patch non-final bytes (no_prev==1) have no previous patch under the paper rule; they run on group 0 for the forward pass and are excluded from the headline aggregate (still counted in the machine-line nonfinal_* cells).

## First-patch exclusion

- no_prev rows: n=1382
- share of all rows: 0.54%
- share of non-final rows: 0.67% (non-final total = 207292)
- no_prev accuracy under the paper rule: 1262/1382 (acc=91.32%)

## Non-final rows — paper rule

Headline rows (is_final==0, no_prev==0) bucketed by patch length, 95% Wilson CI.

| Length | nat | max | bnd |
| --- | --- | --- | --- |
| 1-2 | n=3787 acc=8.61% (7.76%–9.54%) | (no samples) ⚠small-n | n=112 acc=8.04% (4.29%–14.57%) |
| 3-4 | n=51217 acc=12.41% (12.12%–12.69%) | (no samples) ⚠small-n | n=351 acc=16.24% (12.75%–20.46%) |
| 5-8 | n=89280 acc=16.03% (15.79%–16.27%) | (no samples) ⚠small-n | n=551 acc=15.61% (12.82%–18.88%) |
| 9-16 | n=52895 acc=20.99% (20.64%–21.34%) | n=7350 acc=20.97% (20.05%–21.91%) | n=367 acc=20.16% (16.38%–24.57%) |
| 17+ | (no samples) ⚠small-n | (no samples) ⚠small-n | (no samples) ⚠small-n |

## Non-final by closure

ours = headline accuracy under the paper rule; matched baseline = baseline accuracy on the identical (window, row) subset; all-baseline nonfinal = baseline accuracy over ALL its non-final rows of that closure (repo rule); delta = ours − matched baseline.

- nat: ours acc=16.28% (n=197179) | matched baseline acc=98.58% (n=197179) | all-baseline nonfinal acc=98.54% (n=198546) | delta -82.31pp
- max: ours acc=20.97% (n=7350) | matched baseline acc=91.41% (n=7350) | all-baseline nonfinal acc=91.38% (n=7365) | delta -70.45pp
- bnd: ours acc=16.36% (n=1381) | matched baseline acc=97.03% (n=1381) | all-baseline nonfinal acc=97.03% (n=1381) | delta -80.67pp
- overall: ours acc=16.44% (n=205910) | matched baseline acc=98.32% (n=205910) | all-baseline nonfinal acc=98.27% (n=207292) | delta -81.87pp
- note: all-baseline nonfinal nat reproduces the published 98.54% on 198546 rows (195639/198546 = 98.5359%).

## Position within patch

Headline rows bucketed by 0-based position of the row inside its patch.

| Position | nat | max | bnd |
| --- | --- | --- | --- |
| 0 | n=46305 acc=6.87% (6.64%–7.10%) | n=490 acc=6.12% (4.32%–8.61%) | n=404 acc=5.94% (4.02%–8.69%) |
| 1 | n=42518 acc=16.79% (16.44%–17.15%) | n=490 acc=14.69% (11.83%–18.10%) | n=292 acc=17.81% (13.85%–22.61%) |
| 2 | n=35508 acc=15.32% (14.95%–15.70%) | n=490 acc=11.43% (8.91%–14.55%) | n=211 acc=18.48% (13.83%–24.27%) |
| 3 | n=23109 acc=20.00% (19.49%–20.52%) | n=490 acc=20.41% (17.08%–24.20%) | n=148 acc=17.57% (12.28%–24.50%) |
| 4-7 | n=39492 acc=21.95% (21.55%–22.36%) | n=1960 acc=17.30% (15.69%–19.03%) | n=271 acc=24.72% (19.96%–30.19%) |
| 8-15 | n=10247 acc=29.69% (28.81%–30.58%) | n=3430 acc=27.52% (26.05%–29.04%) | n=55 acc=32.73% (21.81%–45.90%) ⚠small-n |
| 16+ | (no samples) ⚠small-n | (no samples) ⚠small-n | (no samples) ⚠small-n |

## Invariant check

no_prev==1 rows keep group 0 under both conventions and sit at the start of a window, where nothing before them changes; their correct flags must be identical between this dump and the baseline (hard invariant). final_flips are informational: rows with is_final==1 keep group j, but the decoder's full-causal byte self-attention mixes earlier non-final rows' changed cross-attn outputs into later rows, so final-row logits shift slightly without their own group changing.

- joined rows: 256000 (ours 256000 / baseline 256000, expected 256000)
- final_flips: 2171 (informational: causal self-attention mixing from earlier changed rows)
- no_prev_flips: 0 (must be 0) — OK
- nonfinal_changed_flips: 169464 (informational: headline rows whose group id changed under the paper rule)

## Small-n caveats

Cells below n=100 are unstable and flagged ⚠small-n above:

- headline/nat/len 17+: n=0
- headline/nat/pos 16+: n=0
- headline/max/len 1-2: n=0
- headline/max/len 3-4: n=0
- headline/max/len 5-8: n=0
- headline/max/len 17+: n=0
- headline/max/pos 16+: n=0
- headline/bnd/len 17+: n=0
- headline/bnd/pos 8-15: n=55
- headline/bnd/pos 16+: n=0
