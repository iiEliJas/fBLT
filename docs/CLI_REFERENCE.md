# CLI Reference

All commands assume the repo root as working directory. Build output goes to `build/` (CPU) or `build-cuda/` (CUDA).

---

## Build & Test

| Command | Description |
|---------|-------------|
| `cmake --build build --target test` | Build + run full C test suite (69 tests, ~4s) |
| `cmake --build build-cuda --target test` | Same, CUDA backend |
| `cmake --build build --target main` | Build minimal main (linking stub) |
| `cmake --build build-cuda --target main` | Same, CUDA |
| `cmake --build build-cuda --target train_blt_d` | Build `train_blt_d` |
| `cmake --build build --target sandbox_run` | Build + run scratch playground |
| `cmake --build build-cuda --target cuda-smoke` | Device sanity check (H2D → kernel → D2H) |
| `cmake --build build-cuda --target cuda-sanitize` | Run tests under `compute-sanitizer` (native Linux) |
| `cmake --build build --target sanity_check` | Build `sanity_check` diagnostic tool |
| `cmake --build build --target last_row_acc` | Build `last_row_acc` diagnostic tool |
| `cmake --build build --target pos_accuracy` | Build `pos_accuracy` diagnostic tool |
| `cmake --build build --target patch_trunc_split` | Build `patch_trunc_split` diagnostic tool |
| `pytest tests/` | Python test suite (config round-trips, YAML, overrides, entry points) |
| `ruff check .` | Python lint |
| `ruff format --check .` | Python format check |

---

## Python Wrapper: `fblt-train`

Installed via `pip install -e .`. Wraps `train_blt_d` with YAML config loading and run-directory management.

```
fblt-train --backend cpu|cuda [--config FILE] [--override key=value ...] [--run-name NAME]
```

| Flag | Required | Description |
|------|----------|-------------|
| `--backend {cpu,cuda}` | yes | Compute backend (never in YAML) |
| `--config FILE` | no | YAML config file (see `configs/train/`) |
| `--override key=value` | no | Override any config field (repeatable) |
| `--run-name NAME` | no | Run directory name (default: `{config}-{timestamp}`) |

The wrapper creates `runs/<run-name>/` with `resolved_config.yaml` containing the full effective config. All `train_blt_d` flags can be set via YAML or `--override`. The `--backend` flag must always be passed on the command line.

```bash
fblt-train --config configs/train/production.yaml --backend cuda
fblt-train --config configs/train/debug.yaml --backend cpu --override steps=100
```

---

## Python Wrapper: `fblt-infer`

Installed via `pip install -e .`. Wraps `infer` with YAML config loading and auto shape-matching.

```
fblt-infer --backend cpu|cuda --checkpoint FILE [--config FILE] [--override key=value ...]
           [--prompt TEXT | --prompt-file FILE] [--output FILE]
```

| Flag | Required | Description |
|------|----------|-------------|
| `--backend {cpu,cuda}` | yes | Compute backend (never in YAML) |
| `--checkpoint FILE` | yes | Model checkpoint (.fblt) |
| `--config FILE` | no | YAML config file (see `configs/infer/`) |
| `--override key=value` | no | Override any config field (repeatable) |
| `--prompt TEXT` | no | Prompt text (mutually exclusive with --prompt-file) |
| `--prompt-file FILE` | no | Prompt file (mutually exclusive with --prompt) |
| `--output FILE` | no | Output file (default: stdout) |

If no `--prompt` or `--prompt-file`, reads from stdin. Auto-detects model dimensions from `resolved_config.yaml` next to the checkpoint (when produced by `fblt-train`).

```bash
fblt-infer --checkpoint runs/debug_checkpoint.fblt --backend cpu --prompt "int main"
fblt-infer --checkpoint my_model.fblt --backend cpu \
  --override embed=192 --override hidden=384 \
  --override enc-layers=2 --override glob-layers=2 --override dec-layers=2
```

---

## Training: `train_blt_d`

**Binary**: `build/train_blt_d` (CPU) / `build-cuda/train_blt_d` (CUDA)

### Core

| Flag | Default | Description |
|------|---------|-------------|
| `--corpus FILE` | *required* | Raw byte corpus |
| `--steps N` | 2000 | Optimizer steps |
| `--lr F` | 0.05 | Learning rate |
| `--block-size B` | 4 | Diffusion block size |
| `--window W` | 48 | Clean-sequence length per example |
| `--embed E` | 64 | Model width |
| `--hidden H` | 128 | FFN width |
| `--layers L` | 2 | Layers per submodule (encoder/global/decoder) |
| `--max-norm F` | 5.0 | Global gradient clip threshold |
| `--d0-mode MODE` | learned | `zeros` \| `learned` for decoder d0 embed |
| `--seed S` | 7 | RNG seed |
| `--report-every K` | 25 | Print frequency |
| `--deterministic` | off | Deterministic training (slower, no cuda atomics) |

### Optimizer

| Flag | Default | Description |
|------|---------|-------------|
| `--optimizer sgd\|adamw` | sgd | Optimizer selection |
| `--beta1 F` | 0.9 | AdamW first moment decay |
| `--beta2 F` | 0.999 | AdamW second moment decay |
| `--eps F` | 1e-8 | AdamW numerical stability |
| `--weight-decay F` | 0.01 | AdamW decoupled L2 weight decay |

SGD uses vanilla gradient descent with global-norm clip at 5.0. AdamW uses the standard PyTorch-compatible algorithm (bias-corrected m/v).

### Mode Selection

| Flag | Default | Description |
|------|---------|-------------|
| `--diffusion 0\|1` | 1 | 0 = plain causal BLT, 1 = BLT-D (masked diffusion) |

### Checkpointing

| Flag | Description |
|------|-------------|
| `--save-weights PATH` | Write weights after training |
| `--save-every N` | Save checkpoint every N steps (0 = only at end) |
| `--load-weights PATH` | Load weights before training (steps=0 → eval only) |
| `--eval-corpus FILE` | Held-out corpus for causal BPB eval |
| `--eval-windows N` | Eval window count (default 200) |
| `--eval-skip N` | Bytes to skip before first eval window |
| `--eval-every N` | Run causal BPB eval every N steps (default 0, disabled) |

### Diffusion Schedule

| Flag | Default | Description |
|------|---------|-------------|
| `--t-min F` | 0.1 | Diffusion timestep floor |
| `--lr-decay 0\|1` | 0 | ×0.3 at 60% and 85% of steps (default schedule) |
| `--lr-decay-steps LIST` | — | Comma-separated step numbers for custom decay points (overrides `--lr-decay`) |
| `--lr-decay-factor F` | 0.3 | Multiplicative factor per custom decay point |
| `--mask-warmup N` | 0 | Ramp L_mask scale 0→1 over N steps |
| `--mask-scale F` | 1.0 | Ceiling for L_mask weight |
| `--mask-late-step N` + `--mask-late-scale F` | 0 | Late ramp from mask_scale to mask_late_scale |
| `--t-warmup-hi F` + `--t-hi-start F` | 0 + 0.8 | High-t curriculum (fraction + start floor) |

### Patching

| Flag | Description |
|------|-------------|
| `--entropy-patches` | Segment with entropy LM + patcher |
| `--train-entropy-lm PATH` | Train standalone entropy LM, save to PATH |
| `--entropy-lm PATH` | Load pretrained entropy-LM weights |

### Backend

| Flag | Default | Description |
|------|---------|-------------|
| `--backend cpu\|cuda` | cpu | Select backend |
| `--cuda-scratch-mb N` | 512 | CUDA scratch arena size in MB (temp allocations during ops) |

### Per-Submodule Overrides

| Flag | Description |
|------|-------------|
| `--enc-layers N` | Encoder layers (default `--layers`) |
| `--glob-layers N` | Global transformer layers |
| `--dec-layers N` | Decoder layers |
| `--cross-attn all\|last` | Cross-attention placement (default all) |

### Logging (diagnostic, all optional)

| Flag | Default | Description |
|------|---------|-------------|
| `--grad-norm-log FILE` | (none) | Write pre-clip gradient norms per step |
| `--update-norm-log FILE` | (none) | Write post-clip update norms at spike steps |
| `--component-norm-log FILE` | (none) | Write per-component gradient norms per step |
| `--activation-dump-log FILE` | (none) | Log activation stats every report-every steps |
| `--batch-log FILE` | (none) | Log batch properties per step |
| `--loss-log FILE` | (none) | Write per-step loss |

---

## build/infer — Production inference

**Note:** For most use cases, prefer `fblt-infer` (Python wrapper above) which handles config loading and shape auto-detection. Use the raw `build/infer` binary directly only when you need to skip the wrapper or debug it.

Single-checkpoint, single-prompt, one-shot generation tool.

```
build/infer --checkpoint FILE --embed E --hidden H --enc-layers N --glob-layers N --dec-layers N --backend cpu|cuda [options]
```

### Required

| Flag | Description |
|------|-------------|
| `--checkpoint FILE` | Model checkpoint (.fblt) |
| `--embed E` | Model embed dim (must match checkpoint) |
| `--hidden H` | Model hidden dim (must match checkpoint) |
| `--enc-layers N` | Encoder layers (must match checkpoint) |
| `--glob-layers N` | Global transformer layers (must match checkpoint) |
| `--dec-layers N` | Decoder layers (must match checkpoint) |
| `--backend cpu\|cuda` | Compute backend |

### Prompt Source

Exactly one required.

| Flag | Description |
|------|-------------|
| `--prompt TEXT` | Raw bytes to use as prompt |
| `--prompt-file FILE` | Read prompt from file (raw bytes) |
| (stdin) | Read raw bytes from stdin until EOF |

### Generation Options

| Flag | Default | Description |
|------|---------|-------------|
| `--new-bytes N` | 64 | Number of bytes to generate |
| `--method MODE` | greedy | `greedy`, `selfspec`, `blockdiff`, `blockdv` |
| `--seed N` | 11 | RNG seed (for random-init entropy LM) |

### Self-Speculation Options

Used with `--method selfspec`.

| Flag | Default | Description |
|------|---------|-------------|
| `--k N` | 8 | Speculative draft window size |

### Block Diffusion Options

Used with `--method blockdiff` or `blockdv`.

| Flag | Default | Description |
|------|---------|-------------|
| `--block-size B` | 8 | Diffusion block size |
| `--unmask STRAT` | confidence | `confidence` or `eb` (entropy-bounded) |
| `--threshold F` | 0.7/1.0 | Alpha (confidence) or gamma (eb) |
| `--boundary-aligned` | off | Commit only at patch boundaries |
| `--adaptive` | off | Adaptive block size |
| `--b-min N` | 4 | Adaptive lower bound |
| `--b-max N` | 16 | Adaptive upper bound |
| `--accept-target F` | 0.5 | Rolling acceptance target |
| `--adapt-window N` | 8 | Rounds per rolling average |

### Model Options

| Flag | Default | Description |
|------|---------|-------------|
| `--cross-attn MODE` | all | `all` or `last` cross-attention placement |

### Patcher Options

| Flag | Default | Description |
|------|---------|-------------|
| `--fixed-patches` | off | Fixed-stride-4 patching (matches training) |
| `--patch-threshold-global F` | 2.5 | Global entropy threshold |
| `--patch-threshold-monotonic F` | 1.0 | Monotonic threshold |
| `--max-patch-length N` | 16 | Maximum patch size |
| `--entropy-lm FILE` | (none) | Trained entropy-LM weights |

### I/O Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output FILE` | stdout | Write generated bytes to file |

### Examples

```bash
# Greedy generation with fixed patches (matches training)
build/infer --checkpoint runs/my_model.fblt \
  --embed 192 --hidden 384 --enc-layers 2 --glob-layers 2 --dec-layers 2 \
  --backend cpu --fixed-patches --prompt "int main()" --new-bytes 64

# BLT-DV with entropy patching
build/infer --checkpoint runs/my_model.fblt \
  --embed 192 --hidden 384 --enc-layers 2 --glob-layers 2 --dec-layers 2 \
  --backend cpu --method blockdv --block-size 8 --threshold 0.7 \
  --entropy-lm runs/entlm.bin --prompt "def " --new-bytes 128

# CUDA inference with output to file
build-cuda/infer --checkpoint runs/my_model.fblt \
  --embed 192 --hidden 384 --enc-layers 2 --glob-layers 2 --dec-layers 2 \
  --backend cuda --fixed-patches --prompt-file prompt.bin \
  --new-bytes 256 --output generated.bin
```

Generated bytes go to stdout (or `--output` file) as raw bytes. All diagnostics and stats go to stderr.

Exit codes: 0 on success, 1 on error.

---

## Inference Benchmark: `infer_bench`

> **Note:** `infer_bench` is a research/benchmarking tool for comparing methods against paired checkpoints. For generating text from a single checkpoint, use `fblt-infer` or `build/infer` instead.

**Binary**: `build/infer_bench` (CPU) / `build-cuda/infer_bench` (CUDA)

| Flag | Default | Description |
|------|---------|-------------|
| `--plain FILE` | runs/plain_40k.fblt | Plain model checkpoint |
| `--bltd FILE` | runs/bltd_l03_40k.fblt | BLT-D model checkpoint |
| `--embed N` | 64 | Embed dim (must match checkpoint) |
| `--hidden N` | 128 | Hidden dim (must match checkpoint) |
| `--dec-layers N` | 2 | Decoder layers (must match checkpoint) |
| `--backend cpu\|cuda` | cpu | Backend |
| `--new-bytes N` | 64 | Bytes to generate per prompt |
| `--dv-gate F` | 0.90 | Min agreement for verified methods (0 = disable) |
| `--fixed-patches` | off | Fixed-stride-4 patching (vs entropy) |
| `--entropy-lm FILE` | off | Trained entropy LM for patching |
| `--results FILE` | bench/results.jsonl | Output results file |

---

## CUDA Benchmark: `cuda_bench`

**Binary**: `build-cuda/cuda_bench`

| Flag | Description |
|------|-------------|
| `--backend cpu\|cuda` | Backend |
| `--iters N` | Iterations per micro-benchmark |
| `--ckpt FILE` | Checkpoint for macro gen (default runs/plain_40k.fblt) |
| `--results FILE` | Output results file |

---

## Diagnostic Tools

Evaluation tools for measuring prediction accuracy on held-out data. All three share common flags.

### Common Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint FILE` | *required* | Model checkpoint (.fblt) |
| `--corpus FILE` | *required* | Held-out byte corpus for evaluation |
| `--window N` | 512 | Sequence length (must match training) |
| `--embed N` | 256 | Embed dim (must match checkpoint) |
| `--hidden N` | 512 | Hidden dim (must match checkpoint) |
| `--enc-layers N` | 2 | Encoder layers (must match checkpoint) |
| `--glob-layers N` | 6 | Global transformer layers (must match checkpoint) |
| `--dec-layers N` | 2 | Decoder layers (must match checkpoint) |
| `--cross-attn all\|last` | all | Cross-attention placement (must match checkpoint) |
| `--diffusion 0\|1` | 1 | 0 = plain BLT, 1 = BLT-D |
| `--num-windows N` | varies | Number of windows to evaluate |
| `--skip N` | 0 | Bytes to skip before first window |
| `--entropy-lm FILE` | (none) | Trained entropy-LM weights for patching |

Build: `cmake --build build --target sanity_check`, `cmake --build build --target last_row_acc`, `cmake --build build --target pos_accuracy`

---

### `sanity_check` — Interior accuracy

Teacher-forced evaluation: feeds real held-out bytes with true history and reports argmax predictions vs actual next bytes. Measures accuracy across all positions (rows 0 through N-1) to verify the model learned the training distribution.

```
build/sanity_check --checkpoint MODEL --corpus FILE [options]
```

**Output:** Mean CE, Mean BPB, Top-1/5/10 accuracy across all positions. Prints sample predictions for first 3 windows.

```bash
build/sanity_check --checkpoint runs/my_model.fblt \
  --corpus data/tinystories/heldout.bin --window 512 --num-windows 50 \
  --entropy-lm runs/entropylm/entropy_lm.fblt
```

---

### `last_row_acc` — Last-row accuracy

Same forward pass as `sanity_check`, but only evaluates prediction accuracy at row N-1 (the byte immediately after the window). Useful for detecting whether the last position receives proper gradient during training.

```
build/last_row_acc --checkpoint MODEL --corpus FILE [options]
```

**Output:** Mean CE, Mean BPB, Top-1/5/10 accuracy at the last position only. Per-sample details for first 10 windows.

```bash
build/last_row_acc --checkpoint runs/my_model.fblt \
  --corpus data/tinystories/heldout.bin --window 512 --num-windows 100 \
  --entropy-lm runs/entropylm/entropy_lm.fblt
```

---

### `pos_accuracy` — Per-position accuracy

Reports top-1 accuracy at every row (0 through N-1) separately. Reveals whether accuracy drops at specific positions (e.g., the last-row cliff from the off-by-one training loss bug).

```
build/pos_accuracy --checkpoint MODEL --corpus FILE [options]
```

**Output:** Per-position accuracy table, plus min/max/mean summary.

```bash
build/pos_accuracy --checkpoint runs/my_model.fblt \
  --corpus data/tinystories/heldout.bin --window 512 --num-windows 10 \
  --entropy-lm runs/entropylm/entropy_lm.fblt
```

---

### `patch_trunc_split` — Patch truncation split diagnostic

Splits interior-row top-1 accuracy by how the containing patch closed: naturally (entropy trigger) vs forced (max_patch_length or buffer boundary). Used to test Hypothesis B from `docs/LAST_ROW_TRAINING_GAP.md` — whether arbitrarily-truncated patches have a representational weakness that more training alone won't fix.

```
build/patch_trunc_split --checkpoint MODEL --corpus FILE [options]
```

**Output:** Three-way accuracy split (natural / max-length capped / buffer-boundary), plus buffer-boundary sub-buckets by patch length (1–2, 3–4, 5–8, 9–16, 17+).

```bash
build/patch_trunc_split --checkpoint runs/tinystories_p7/tinystories_p7.fblt \
  --corpus data/tinystories/heldout.bin --window 512 --num-windows 50 \
  --entropy-lm runs/entropylm/entropy_lm.fblt
```

---

## Sweep Scripts

| Script | Description |
|--------|-------------|
| `tools/run_at_scale_sweep.sh [--backend cpu\|cuda]` | 9-arm at-scale sweep (27 runs, ~12h on CUDA) |
| `tools/run_ablation_sweep.sh [--backend cpu\|cuda]` | Ablation sweep |
| `tools/run_sweep.sh [--backend cpu\|cuda]` | General sweep runner |

---

## Example Commands

### Toy-scale training (plain + BLT-D, SGD)
```bash
cmake --build build-cuda --target train_blt_d

build-cuda/train_blt_d --corpus data/train.bin --eval-corpus data/heldout.bin \
    --eval-windows 300 --steps 40000 --diffusion 0 --lr-decay 1 --seed 7 \
    --save-weights runs/plain_40k.fblt --backend cuda

build-cuda/train_blt_d --corpus data/train.bin --eval-corpus data/heldout.bin \
    --eval-windows 300 --steps 40000 --diffusion 1 --t-min 0.1 \
    --mask-warmup 10000 --mask-scale 0.3 --lr-decay 1 --seed 7 \
    --save-weights runs/bltd_l03_40k.fblt --backend cuda
```

### Toy-scale training with AdamW
```bash
build-cuda/train_blt_d --corpus data/train.bin --eval-corpus data/heldout.bin \
    --eval-windows 300 --steps 40000 --diffusion 1 --t-min 0.1 \
    --mask-warmup 10000 --mask-scale 0.3 --lr-decay 1 --seed 7 \
    --optimizer adamw --lr 0.001 --weight-decay 0.01 \
    --save-weights runs/bltd_adamw_40k.fblt --backend cuda
```

### Inference comparison
```bash
cmake --build build-cuda --target bench-infer

build-cuda/infer_bench --plain runs/plain_40k.fblt --bltd runs/bltd_l03_40k.fblt \
    --backend cuda --new-bytes 64 --results bench/results.jsonl
```

### At-scale sweep (27 runs)
```bash
tools/run_at_scale_sweep.sh --backend cuda
```
