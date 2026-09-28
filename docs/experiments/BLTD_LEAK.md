# BLT-D Block Training Has a Label Leak: Masked Cells Are Solvable by Copying the Clean Twin

**Status: CONFIRMED — root cause identified, no fix implemented.** Under teacher forcing, BLT-D's
masked-cell prediction is **100% accurate in the training layout** and **3.75% accurate in the
inference layout**, using identical weights and identical code. The gap is a label leak: the
training batch construction places every block cell at a position where a clean row already holds
that cell's target, and block rows can attend it. Gradient descent found the trivial solution, so
`L_mask` trained to 1.0 while teaching nothing about denoising. Inference has no such row.

Consequence: `blt_generate_greedy_blockdiff` emits near-uniform noise including non-UTF-8 bytes, and
BLT-DV acceptance sits at ~1%. This is a retrain-from-scratch fix, not a code tweak.

Symptom reproduced on `runs/tinystories_840k/tinystories.fblt` with `runs/entropylm/entropy_lm.fblt`,
prompt `"Once upon a time"`, 64 new bytes, CPU:

```
greedy   : Once upon a time war a lar tas - a lar tas - a big yar a lar - ...
blockdiff: Once upon a time el nce tn???? ac?""" ?.l;???.n? .. g.""?. .ce.U?...
                                          nfe_enc_glob=8, nfe_dec=30, accepted=64 (100.0%)
blockdv  : ...                                                nfe_dec=365, accepted=5 (1.1%)
```

Garbage starts at the very first block byte and never recovers.

## Origin

BLT-S (`SELFSPEC_BUG.md`) was clean while BLT-D produced noise, and BLT-DV acceptance (~1%) sat near
the 0.39% chance floor for a 256-symbol vocabulary — statistically decorrelated from the AR model's
own predictions rather than merely lower quality. Since drafting and verification share weights,
that pointed at the masked path specifically. Investigation ran cheapest-check-first: unmasking
selection, MASK index, cross-attention convention, RoPE positions, then teacher-forced accuracy.

## Steps 1-2: clean

**Unmasking selection (step 1) is correct.** `blt_unmask_select_confidence`
(`csrc/infer/block_generation.c:24`) tracks the maximum `p_max` — highest confidence first, not
lowest. `blt_unmask_select_eb` (`:37`) is a textbook *ascending* insertion sort on entropy, so
`order[0]` is the lowest-entropy cell, and the cumulative-entropy budget walks that prefix. Neither
operator is flipped.

**MASK embedding index (step 2) is correct.** `BLT_MASK_TOKEN_ID` is `256` (`block_diffusion.h:34`),
the table is `BLT_D0_VOCAB = 257` rows, and the lookup path applies no modulo or clamp. The table is
zeroed each step (`train_blt_d.c:415`), included in the optimizer (`train_optim.c:89`), and loaded
untouched when a checkpoint is supplied.

## Step 3: a real asymmetry, but falsified

There *is* a train/inference asymmetry here, and it was the leading suspect. Training sets
`out->groups[r] = i - 1` for the block belonging to patch `i` (`block_diffusion.c:93`), so block rows
cross-attend only `o_0 .. o_{M-2}`. Latent `o_{M-1}` is never a block-row cross-attention target in
training. Inference sets `groups[b] = num_patches - 1` for every block row
(`block_generation.c:205`), i.e. exclusively `o_{M-1}` — the one latent never trained for that role.

The design intends this: a block follows the prefix, so "the last available latent" `o_M` is
semantically the `o_{i-1}` its training analogue would use. Measured directly, it does not matter:

| t | conv | masked_acc | n |
|---|---|---|---|
| 1.00 | train groups (`o_{i-1}`) | 0.9999 | 19062 |
| 1.00 | infer groups (`o_M`) | 0.9998 | 19062 |

A 0.0001 delta over 19k samples. The leak is reachable through self-attention, so it dominates and
hides any cross-attention effect. **Not the bug.**

## Step 4/root cause: the clean-twin leak

`blt_block_batch_build_t` builds blocks from patches 1..M-1, i.e. from *inside* the clean window:

```c
for (size_t i = 1; i < num_patches; i++) {          // csrc/models/block_diffusion.c:82
    const size_t s = patches[i].start_idx;
    for (size_t k = 0; k < B; k++, r++) {
        const size_t pos = s + k;
        const bool valid = pos < N;                  // :86
        out->cell_valid[r]  = valid ? 1 : 0;
        out->positions[r]   = valid ? pos : (N - 1); // :88
        out->targets[r]     = valid ? bytes[pos] : 0;
```

Every valid block cell therefore satisfies `positions[r] < N`, so a **clean row exists at the same
position**. And the self-attention mask lets block rows see all of them:

```c
allowed = (j < N) || ((j - N) / B <= (i - N) / B);   // csrc/ops/mask_builder_cpu.c:135
```

Block row at position `p` can attend clean row `p`, whose `D_0` is the encoder's `h_final[p]` —
which contains `bytes[p]`, **exactly that cell's target**. The masked-diffusion task reduces to a
copy. The CUDA mask kernel (`mask_builder.cu:172`) is identical, so the CUDA training run has the
same property.

Inference is structurally different. `blt_draft_block` sets `pos[b] = prefix_len + b`
(`block_generation.c:201`), all `>= N`, so no clean twin exists at those positions. **The only block
cells that ever reach `pos >= N` during training are PAD cells, and those are excluded from
`L_mask`** (`cell_masked` requires `cell_valid`, `block_diffusion.c:92`). The model is never trained
on a single byte it must denoise without a twin.

Fast-BLT 3.2.1 specifies `x' = [x_1..x_N ; x_block^t]` — blocks are the *future* region. The repo's
construction instead recycles patches from inside the window, which is a documented deviation
(`block_diffusion.h:62-68`) and is what makes the task degenerate.

### The arithmetic that gives it away

The masked task receives strictly *less* information than the clean AR task, so it cannot
legitimately outscore it:

- Clean row `p` predicts `bytes[p+1]` from `bytes[0..p+1]` → **74.8%**
- Block row `p` predicts `bytes[p]`, with the row containing `bytes[p]` visible to it → **100%**

A strictly harder task scoring 25 points higher is a reachable label, not a prediction.

## Step 5: teacher-forced masked accuracy

Measured with `csrc/tools/masked_acc.c` on `data/tinystories/heldout.bin`, window 256, block size 4,
100 windows, entropy patching (matching training). Teacher-forced, no autoregressive generation.

| t | conv | masked_acc | n | clean_acc | masked p_max | masked H |
|---|---|---|---|---|---|---|
| 0.50 | train | **1.0000** | 9546 | 0.7478 | 1.000 | 0.00 |
| 0.50 | infer | 0.9999 | 9546 | 0.7478 | 1.000 | 0.00 |
| 0.50 | novel | **0.0375** | 400 | 0.7478 | 0.715 | 1.02 |
| 1.00 | train | 0.9999 | 19062 | 0.7478 | 1.000 | 0.00 |
| 1.00 | infer | 0.9998 | 19062 | 0.7478 | 1.000 | 0.00 |
| 1.00 | novel | 0.0375 | 400 | 0.7478 | 0.715 | 1.02 |

`train` / `infer` differ only in the cross-attention group convention. `novel` is the same weights
and the same forward code with the block placed at positions `N..N+B-1` — the inference layout, with
no clean twin.

Chance for 256 symbols is 0.0039. So the inference-layout figure of **3.75%** is ~9.6x chance but
roughly 20x worse than the model's own AR ability, and its entropy (1.02 nats) shows it is
confidently wrong rather than uniform.

### Note on sample size

An earlier run with only 3 windows gave `novel = 0.1250` (n=12, then n=48). Widening to n=400 moved
it to 0.0375. The small-sample figure was ~3x optimistic; treat n=400 as the number. The
`twin - novel` delta is stable regardless (-0.87 to -0.96 across every sample size tried), so the
conclusion never depended on the exact value.

### Patcher-independent

The leak lives in `blt_block_batch_build_t`, which consumes whatever patches it is handed, so both
segmentation paths leak identically. Confirmed, block size 8, 20 windows, fixed-stride patching:

| t | conv | masked_acc | n | clean_acc | masked p_max | masked H |
|---|---|---|---|---|---|---|
| 1.00 | train | 1.0000 | 10000 | 0.5529 | 1.000 | 0.00 |
| 1.00 | infer | 1.0000 | 10000 | 0.5529 | 1.000 | 0.00 |
| 1.00 | novel | **0.0875** | 160 | 0.5529 | 0.737 | 0.90 |

The `fixed_stride` path is audited. (Its lower `clean_acc` of 0.5529 vs 0.7478 is expected — coarse
stride-4 patching is a weaker encoder front-end, which is why training uses entropy patching. The
840k checkpoint was trained with `entropy_patches: true`.)

## Consequence for the reported metrics

- **`masked acc > 0.99` in the README ablation table is this leak, not a capability.** It should be
  relabelled or withdrawn; it cannot be reproduced in a leak-free regime.
- **BLT-DV acceptance ~1%** is the honest consequence: the model has no denoising ability to
  verify against, so draft and AR predictions are close to decorrelated.
- The 1.0 forward-loss values late in `runs/tinystories_840k/loss.log` are not evidence of a
  converged diffusion objective.

## Fix direction (not implemented)

Rebuild blocks from bytes *after* the clean prefix, as Fast-BLT 3.2.1 specifies, so block positions
are `>= N` and no clean twin exists. The self-attention rule then needs no special case: with blocks
strictly past the prefix, `j < N` no longer exposes the answer.

Two things to settle while doing it:

1. **Number of blocks.** With `M` patches and blocks past the prefix, the window only has `N`
   bytes of headroom, so `M-1` blocks of `B` bytes no longer fit. Either cap `M` so
   `N + B*(M-1) <= N` is not needed and blocks are drawn from a suffix, or extend the window. The
   current code gets "enough blocks" by reusing in-window patches, which is precisely the bug.
2. **Overlapping blocks.** Whenever a patch is shorter than `B` (patches average 4-6 bytes at
   `B=4`), consecutive blocks overlap and duplicate RoPE positions. Moving past the prefix does not
   fix this by itself; blocks need to be laid out at `B`-byte strides or the stride needs to be
   `>= B`.

Defensively, forbid block rows from self-attending to clean rows at positions `>=` their own, so a
future regression cannot silently re-open the leak. The cheapest guard against recurrence is a
regression test asserting that masked-cell accuracy *in the inference layout* stays above the clean
AR accuracy's neighbourhood — the current twin-layout measurement cannot detect this class of bug
at all, because the leak makes it saturate.

## Tooling notes

`masked_acc` is a new diagnostic (CMake target). Two defects in it were found and fixed while
producing the numbers above, both worth knowing if the tool is kept:

- The original headroom check `window + block_size * 126 <= 1024` used worst-case `M = 127`
  unconditionally, so it rejected valid configs such as `--window 256 --block-size 8`. It is now
  patcher-aware: exact for `fixed` (where `M = ceil(window/4)` is deterministic), and an exact
  per-window `S = window + B*(M-1)` check for `entropy` right after segmentation.
- `fixed_stride` writes `ceil(window/4)` entries with no internal cap, so `--window 513+` with
  `--patcher fixed` would overrun the 128-entry `patches` array. Guarded.

## Reproducing

```bash
cmake --build build --target masked_acc -j$(nproc)

CKPT=runs/tinystories_840k/tinystories.fblt
ELM=runs/entropylm/entropy_lm.fblt
COMMON="--checkpoint $CKPT --entropy-lm $ELM --corpus data/tinystories/heldout.bin \
        --embed 256 --hidden 512 --enc-layers 2 --glob-layers 6 --dec-layers 2 --cross-attn all \
        --backend cpu --window 256 --layout novel"

# Entropy patching (training config), wide sample
./build/masked_acc $COMMON --windows 100 --block-size 4 --t 0.5 --t 1.0

# Fixed-stride patching, same leak
./build/masked_acc $COMMON --windows 20 --block-size 8 --t 1.0 --patcher fixed
```

Each run takes 7-25 minutes on CPU. `masked_acc` flags: `--patcher entropy|fixed`,
`--layout twin|novel`, repeatable `--t F`, `--block-size`, `--windows`, `--window`, `--t-min`,
`--d0 learned|zeros`, `--seed`. The `VERDICT` line compares `train` against `novel`; the
`infer - train` delta is the step-3 cross-attention check and should stay near zero.
