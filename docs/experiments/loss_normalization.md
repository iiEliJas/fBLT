# BLT-D Loss Normalization

Investigation of how `train_blt_d`'s objective relates to Fast-BLT Eq. 5-7, and
what the `--paper-loss` / `--mask-loss-norm` flags actually change.

## What the paper specifies

Eq. 5 (clean), Eq. 6 (masked), Eq. 7 (total), all as bare sums:

```
L_clean = -sum_{i=1}^{N} log p(x_i | x_<i)
L_mask  = -(1/t) sum_{i=2}^{M} sum_{k=0}^{B-1} 1[b^t_{i-1,k} = [MASK]] log p(x_{s_i+k} | ...)
L_total = L_clean + L_mask
```

## What the repo does

| | paper | repo default |
|---|---|---|
| `L_clean` (Eq. 5) | sum over `N` | **mean** over `clean_rows` |
| `L_mask` (Eq. 6) | `(1/t) * sum` over masked cells | `mask_scale * (1/t) * sum` |
| total (Eq. 7) | `L_clean + L_mask` | same, weighted by `mask_scale` |

Two things are frequently assumed to be deviations but are **not**:

- **The `1/t` factor is already paper-exact.** The repo computes
  `loss += loss_scale * l_mask / batch->t` where `l_mask` is a sum over masked
  cells (`csrc/models/block_diffusion.c`). That is Eq. 6 exactly when
  `loss_scale == 1.0`, which is the batch-construction default.
- **The sum over masked cells is also paper-exact.** Eq. 6 sums; the repo sums.

The single code deviation is that **`L_clean` is a mean**: `blt_cross_entropy_forward`
divides by `seq_len` (`csrc/ops/reductions_cpu.c`). Everything else is a
`mask_scale` weight, which is a config choice rather than a code difference.

`--paper-loss 1` scales `L_clean` by `clean_rows`, making both terms the paper's
sums. Verified in isolation (`--diffusion 1 --mask-scale 0`, `N=128`):
mean `5.5432` -> sum `709.52`, a ratio of exactly `128.0 = clean_rows`.

## The real consequence: clean:mask balance

Because `L_clean` becomes a sum while `L_mask` stays a sum, the balance shifts
by roughly `N`. On the 600k run (`L_clean` mean 2.90, 128 masked cells,
`t~0.5`, `mask_scale` 0.3):

| | `L_clean` term | `L_mask` term | ratio |
|---|---|---|---|
| repo default | 2.90 | 217.0 | 74.8 : 1 favoring mask |
| paper rule | 1484.9 | 723.4 | 0.49 : 1 favoring clean |

So switching is not a rescaling; it inverts what the model spends capacity on.
A paper-faithful `L_total` at that quality is ~2209, versus the ~220 the repo
prints, so **`avg_loss` is not comparable to a paper number either way**.

Note the paper's `L_clean` sum also scales with `--window`, so changing the
window length silently rescales the loss and shifts the clean:mask balance. The
mean form avoids that footgun.

## Gradient norms and why they matter for Adam

`grad_norm.log` records the **pre-clip** norm. On the 600k run
(`--max-norm 5.0`, 225 samples):

| | value |
|---|---|
| min / median / max pre-clip | 13.0 / 1224 / 6651 / 223862 |
| fraction above the clip | 100% |
| median clip factor | 245x |
| max clip factor | 44772x |

Adam is scale-invariant only under a *constant* rescale: if every step were
scaled by the same `c`, `m/sqrt(v)` is unchanged and clipping is harmless. Here
the clip factor ranges over 2.6x-44772x, so `v` accumulates `g^2` at wildly
inconsistent scales and the moment estimates degrade. That is the likely
mechanism behind the loss swings observed between adjacent 2k checkpoints.

Toy-run measurements (40 steps, `--optimizer adamw`, `--report-every 10`):

| mode | median pre-clip grad norm | max/min spread |
|---|---|---|
| default (`L_clean` mean) | 134961 | 40.3x |
| `--paper-loss 1` | 197805 | **2.5x** |
| `--mask-loss-norm 1` (toy, 60 steps) | 1201 | **5.6x** |

The spread matters more than the magnitude. The paper form is far closer to a
constant rescale, so **clipping becomes largely harmless under it** -- which
resolves the instability rather than worsening it. That is the main argument
for adopting it.

## Recommendation

1. Use `--paper-loss 1` for fidelity; it also happens to be the more
   Adam-stable option.
2. Leave `--mask-scale 1.0`. The `0.3` used in the repo's toy configs is a
   stabilization, not the paper, and `mask_scale != 1.0` means `L_total` is not
   Eq. 7.
3. **Re-derive `--max-norm`.** Paper-exact gradients are ~200k, so any clip
   threshold inherited from a mean-normalized loss is meaningless. Either raise
   it to ~1e5, or drop global clipping and rely on Adam's own normalization,
   since the near-constant clip factor under the paper form means the clip is
   close to a no-op anyway.
4. Ablate `mask_scale` (0.3 vs 1.0) separately before concluding, since it
   changes the clean:mask balance independently of the normalization.

## Caveats

- All numbers above are from small/toy runs except the 600k gradient norms,
  which come from a single configuration and vary with patch density per window.
- `--paper-loss 1` changes gradient scale by ~`N`. Any learning rate tuned
  against the mean form is far too large for it.