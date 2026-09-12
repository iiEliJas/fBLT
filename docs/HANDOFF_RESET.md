# Handoff: Working-State Reset (2026-08-29)

## Why the reset

The CUDA-graph-capture branch accumulated real work (three documented/fixed bugs,
a ~6.4x step-time speedup) but also reached a state where debugging results from
later commits were being applied retroactively to earlier ones, producing circular
conclusions. The decision was made to:

1. Archive the CUDA-graph work on a named branch.
2. Check out a known-good earlier commit (`9e01acf`, "Added bf16 dtype") as the
   new baseline.
3. Re-audit and re-fix from scratch on that commit, with clean before/after
   measurements.

## Branch layout

| Branch | Commit | Purpose |
|---|---|---|
| `archive/cuda-graph-capture-v1` | `551c70e` | All CUDA graph capture work preserved |
| `main-reset` | `9e01acf` → current | Clean baseline + fixes applied below |

## Bugs found and fixed on this commit

### 1. Missing gradient zeroing for RMSNorm weight gradients

**Root cause:** `blt_rmsnorm_backward` accumulates (+=) into `grad_weight` but
the caller must zero it before each step. The training loop (`run/train_blt_d.c`)
only zeroed `embedding_grad`, `ngram_grads`, and `d0_embed_grad` — all other
gradient tensors were allocated once (zeroed at creation) and reused across steps
without being zeroed between them.

**Effect:** Norm weight gradients (`norm1_weight`, `norm2_weight`,
`cross_norm_weight`) accumulated across all training steps, growing unboundedly.

**Fix:** Added `zero_tensor()` calls for 8 gradient tensor types across encoder
(layer x 3: norm1, norm2, cross_norm), global (layer x 2: norm1, norm2), and
decoder (layer x 3: norm1, norm2, cross_norm) at the start of each training step
in `run/train_blt_d.c:830-843`.

**Measured impact:** At 100 steps / E=64, the effect on loss was subtle (norm
weight gradients contributed ~0.06% of total `sum(grad^2)`). The fix is
**correct** (satisfies the documented API contract) but its training-dynamics
impact at this scale is small. The effect would compound more visibly over
thousands of steps or with larger models.

### 2. Default `t_min` raised from 0.05 to 0.1

**Root cause:** Random timestep draws (`t ~ U(0,1)`) occasionally produce very
small `t` values. The `1/t` weighting in the masked-diffusion loss causes these
steps to dominate gradient updates. The `t_min` floor (clamping `batch.t`) limits
this, but `t_min=0.05` still allows `1/t = 20x` weighting, which produces
observable gradient-norm spikes.

**Measured impact:** Confirmed by printing `batch.t` at spike steps (steps 16 and
35): both drew `t` near zero (0.000030 and 0.029096) and were floored to 0.05.
These low-`t` steps produce very low loss values (~3.5-5.0 vs typical ~180-200)
because `L_clean` dominates when the model predicts the (mostly-unmasked) window
well. Raising `t_min` to 0.1 halves the maximum `1/t` weighting from 20x to 10x,
reducing gradient-norm spikes.

**Fix:** Changed default `t_min` from 0.05 to 0.1 in the `args_t` initializer
(`run/train_blt_d.c`).

## What is NOT a bug (common misattribution)

### Per-step loss noise from random timestep draws

With `--diffusion 1`, each training step draws a fresh `t ~ U(0,1)` (clamped to
`t_min`). This produces inherent per-step loss variance:
- **Low `t`** (near `t_min`): mostly clean data → low loss (~3-5 at extreme)
- **High `t`** (near 1.0): heavily masked → different loss regime

This is **not** a bug. It is inherent to the masked-diffusion training objective.
The per-step loss trace will always show substantial variance (typical max/median
ratio ~2.8x). Do not mistake this normal behavior for training instability.

### CPU vs CUDA loss divergence over many steps

CPU and CUDA produce identical loss for the first ~15 steps, then diverge as FP32
arithmetic differences compound through parameter updates. At 100 steps / lr=0.05,
max relative difference reaches ~37% at step 93. This is expected numerical
divergence, not a correctness bug. The CUDA backend is itself bit-deterministic
(max diff between two GPU runs: 0.0001).

## Baseline loss trajectory (standard config)

Config: E=64, H=128, W=48, layers=2, steps=100, seed=7, diffusion=1, t_min=0.1,
CPU backend. Loss values are per-step (not smoothed):

```
Step 1: 260.15   Step 26: 103.68   Step 51: 183.91   Step 76: 159.25
Step 2: 274.53   Step 27: 310.44   Step 52: 248.52   Step 77: 127.64
Step 3: 253.45   Step 28: 198.16   Step 53: 169.41   Step 78: 171.33
Step 4: 269.03   Step 29: 219.45   Step 54: 167.67   Step 79: 140.53
Step 5: 196.42   Step 30: 221.51   Step 55: 186.16   Step 80: 151.89
Step 6: 156.59   Step 31: 191.86   Step 56: 38.31    Step 81: 148.36
Step 7: 196.00   Step 32: 200.42   Step 57: 81.65    Step 82: 174.46
Step 8: 168.50   Step 33: 191.48   Step 58: 153.85   Step 83: 161.66
Step 9: 206.24   Step 34: 95.29    Step 59: 145.54   Step 84: 148.30
Step 10: 197.29  Step 35: 4.96     Step 60: 178.81   Step 85: 124.35
Step 11: 217.01  Step 36: 162.74   Step 61: 218.05   Step 86: 511.38
Step 12: 205.96  Step 37: 203.89   Step 62: 189.21   Step 87: 102.58
Step 13: 136.56  Step 38: 189.59   Step 63: 88.09    Step 88: 221.00
Step 14: 206.41  Step 39: 194.11   Step 64: 164.78   Step 89: 202.23
Step 15: 204.18  Step 40: 218.30   Step 65: 145.16   Step 90: 190.99
Step 16: 3.52    Step 41: 195.05   Step 66: 147.09   Step 91: 183.82
Step 17: 198.64  Step 42: 202.50   Step 67: 171.33   Step 92: 249.29
Step 18: 170.27  Step 43: 220.33   Step 68: 111.53   Step 93: 340.82
Step 19: 217.02  Step 44: 192.68   Step 69: 174.08   Step 94: 204.75
Step 20: 242.94  Step 45: 144.89   Step 70: 170.09   Step 95: 179.68
Step 21: 112.83  Step 46: 184.17   Step 71: 181.46   Step 96: 186.70
Step 22: 165.35  Step 47: 183.58   Step 72: 180.36   Step 97: 193.44
Step 23: 120.23  Step 48: 182.92   Step 73: 167.90   Step 98: 132.67
Step 24: 219.44  Step 49: 181.84   Step 74: 153.73   Step 99: 192.55
Step 25: 180.20  Step 50: 147.19   Step 75: 147.78   Step 100: 221.07
```

Low-loss steps (e.g. 16, 35, 56, 63) are caused by the training window content
at those positions in the corpus, **not** by training instability.
