# Training Ablations

Architecture and schedule ablations at 2.97M parameters (embed=192, hidden=384, 2/2/2 layers) on ~67 MB of C source code. Each axis tested with 3-6 seeds per configuration at 40k steps.

## Production config

Confidence tags: **CONFIRMED** = multi-seed validated. **PROVISIONAL** = limited evidence or untested.

```
--embed 192 --hidden 384 --layers 2
--steps 40000 --lr 0.05 --lr-decay 1
--mask-warmup 33200 --mask-scale 0.3
--t-min 0.1 --t-warmup-hi 0.25 --t-hi-start 0.8
--cross-attn all --optimizer sgd
--diffusion 1 --block-size 4 --window 48
--d0-mode learned
```

| Parameter | Value | Confidence | Note |
|---|---|---|---|
| Steps | 40000 | CONFIRMED | Safe convergence floor; masked acc > 0.99 |
| Mask-warmup | 33200 (83%) | CONFIRMED | 95% is a minor free improvement, not required |
| Mask-scale | 0.3 | CONFIRMED | Standard across all experiments |
| Cross-attn | all (base) or last (xlast) | CONFIRMED | xlast is ~15% faster, quality indistinguishable |
| High-t warmup | --t-warmup-hi 0.25 --t-hi-start 0.8 | CONFIRMED | -0.33 BPB improvement on base |
| Mask-late | off (default) | CONFIRMED | No benefit found |
| Encoder layers | 2 | PROVISIONAL | n=4, single model size; deeper encoder shows instability |
| Decoder layers | 2 | CONFIRMED | Deeper decoder doesn't help |
| Global layers | 2 | CONFIRMED | Matches decoder/encoder depth |
| LR | 0.05 | CONFIRMED | Pre-investigation finding; validated across 4 seeds |
| LR-decay | x0.3 at 60%/85% | CONFIRMED | Pre-investigation finding; fraction-based |
| Optimizer | SGD | PROVISIONAL | Never tested as a variable at real scale |
| Block size | 4 | PROVISIONAL | Assumed default; never tested as a variable |
| Window | 48 | PROVISIONAL | Assumed default; never tested as a variable |
| d0-mode | learned | PROVISIONAL | Assumed default; never tested as a variable |
| t-min | 0.1 | CONFIRMED | Standard across all experiments |
| Embed | 192 | CONFIRMED | Standard across all experiments |
| Hidden | 384 | CONFIRMED | Standard across all experiments |

## Axis results

### Decoder depth

| Split (enc/glob/dec) | Params | Mean BPB | Mean Acc | Confidence |
|---|---|---|---|---|
| 2/2/2 | 2.97M | 3.033 | 0.993 | CONFIRMED |
| 2/2/3 | 3.49M | 3.354 | 0.989 | CONFIRMED |
| 2/2/4 | 4.00M | 3.173 | 0.987 | CONFIRMED |
| 3/2/2 | 3.49M | 4.219 | 0.854 | CONFIRMED |

**Finding:** Deeper decoder (2/2/3, 2/2/4) doesn't help. Even with the +18-34% params, no quality gain. The 2/2/2 baseline is optimal. Deeper encoder (3/2/2) is harmful: +1.186 BPB worse, 56x more acc variance, seed-dependent convergence failure.

**Note:** Encoder depth result is PROVISIONAL. Only done with 4 seeds, single model size (2.97M), not tested at other step budgets. The BLT paper §7/Table 9 reports the same pattern: "shallower encoder paired with deeper decoder consistently outperforms the reverse."

### Cross-attention placement

| Config | Cross-Attn | Warmup | Mean BPB | Mean Acc | Wall (s) |
|---|---|---|---|---|---|
| base | all | 83% | 3.033 | 0.993 | ~791 |
| xlast | last | 83% | 2.898 | 0.998 | ~668 |
| xlast + high-t | last | 83% | 2.794 | 0.998 | ~668 |
| base + high-t | all | 83% | 2.703 | 0.997 | ~791 |

**Finding:** xlast's earlier schedule (mask-warmup 0) doesn't converge at this scale: at 0% warmup, xlast is unstable (acc 0.898, 1/4 seeds fail). At 83% warmup, xlast converges as good as base. Quality is the same for base and xlast at matched warmup. xlast is ~15% faster per run.

**Practical recommendation:** If speed matters, pick xlast. If safety matters, pick base. Both work.

### High-t warmup

| Config | High-t | Mean BPB | BPB Std | Mean Acc |
|---|---|---|---|---|
| base baseline | off | 3.033 | 0.124 | 0.993 |
| base + high-t | on (0.25/0.8) | 2.703 | 0.129 | 0.997 |

**Finding:** High-t warmup improves BPB by 0.33 (2.7x the noise floor) with no convergence cost. All seeds improve. On xlast, the improvement is smaller (-0.105 BPB, within noise).

### Mask-late schedule

| Config | Mask-late | Mean BPB | Mean Acc |
|---|---|---|---|
| off (baseline) | disabled | 3.033 | 0.993 |
| on | 14000 / 1.0 | 3.007 | 0.994 |

**Finding:** Null result. BPB delta (-0.03) is 0.2x the noise floor, within random seed variation. No convergence cost, but no benefit either. Dont include in production config.

### Warmup fraction (at fixed 30k steps)

| Fraction | Mean BPB | Mean Acc |
|---|---|---|
| 50% | 4.927 | 0.485 |
| 65% | 4.758 | 0.635 |
| 83% | 4.666 | 0.751 |
| 95% | 4.616 | 0.825 |

**Finding:** Higher warmup fraction gives better convergence (monotonic in the tested range) 95% beats 83% by +0.07 acc and half the variance, but the absolute gap is modest. 83% is the validated default; 95% is a minor free improvement.

### Step budget

| Steps | Mean BPB | Mean Acc |
|---|---|---|
| 20k | 5.356 | 0.208 |
| 25k | 5.192 | 0.349 |
| 30k | 4.666 | 0.751 |
| 35k | 3.677 | 0.910 |
| 40k | 3.033 | 0.993 |

**Finding:** Sharp cliff between 25k-35k steps. 40k is the first cell with reliable convergence (acc > 0.99, std 0.002). The cliff sits somewhere in the 25k-35k window.

## Determinism

CUDA non-determinism (atomicAdd-based scatter_add, cuBLAS parallel reductions) causes run-to-run variance. A `--deterministic` flag exists (single-threaded scatter_add + CUBLAS_PEDANTIC_MATH) for debugging and ablation sweeps, but costs too much throughput for production training. So same seed + same config does not guarantee identical results without this flag.
