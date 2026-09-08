# CLI Reference

All commands assume the repo root as working directory. CUDA builds output to `bin-cuda/` and `obj-cuda/`; CPU builds to `bin/` and `obj/`.

---

## Build & Test

| Command | Description |
|---------|-------------|
| `make test` | Build + run full test suite (68 tests, ~4s) |
| `make CUDA=1 test` | Same, CUDA backend |
| `make main` | Build minimal main (linking stub) |
| `make CUDA=1 main` | Same, CUDA |
| `make CUDA=1 train-blt-d` | Build `train_blt_d` |
| `make sandbox` | Build + run scratch playground (`run/sandbox.c`) |
| `make cuda-smoke` | Device sanity check (H2D → kernel → D2H) |
| `make cuda-sanitize` | Run tests under `compute-sanitizer` (native Linux) |

---

## Training: `train_blt_d`

**Binary**: `bin/train_blt_d` (CPU) / `bin-cuda/train_blt_d` (CUDA)

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
| `--load-weights PATH` | Load weights before training (steps=0 → eval only) |
| `--eval-corpus FILE` | Held-out corpus for causal BPB eval |
| `--eval-windows N` | Eval window count (default 200) |
| `--eval-skip N` | Bytes to skip before first eval window |

### Diffusion Schedule

| Flag | Default | Description |
|------|---------|-------------|
| `--t-min F` | 0.1 | Diffusion timestep floor |
| `--lr-decay 0\|1` | 0 | ×0.3 at 60% and 85% of steps |
| `--mask-warmup N` | 0 | Ramp L_mask scale 0→1 over N steps |
| `--mask-scale F` | 1.0 | Ceiling for L_mask weight |
| `--mask-late-step N` + `--mask-late-scale F` | 0 | Late ramp from mask_scale to mask_late_scale |
| `--t-warmup-hi F` + `--t-hi-start F` | 0 | High-t curriculum |

### Patching

| Flag | Description |
|------|-------------|
| `--entropy-patches` | Segment with entropy LM + patcher |
| `--train-entropy-lm PATH` | Train standalone entropy LM, save to PATH |
| `--entropy-lm PATH` | Load pretrained entropy-LM weights |

### Backend

| Flag | Description |
|------|-------------|
| `--backend cpu\|cuda` | Select backend |

### Per-Submodule Overrides

| Flag | Description |
|------|-------------|
| `--enc-layers N` | Encoder layers (default `--layers`) |
| `--glob-layers N` | Global transformer layers |
| `--dec-layers N` | Decoder layers |
| `--cross-attn all\|last` | Cross-attention placement (default all) |

---

## Inference Benchmark: `infer_bench`

**Binary**: `bin/infer_bench` (CPU) / `bin-cuda/infer_bench` (CUDA)

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

**Binary**: `bin-cuda/cuda_bench`

| Flag | Description |
|------|-------------|
| `--backend cpu\|cuda` | Backend |
| `--iters N` | Iterations per micro-benchmark |
| `--ckpt FILE` | Checkpoint for macro gen (default runs/plain_40k.fblt) |
| `--results FILE` | Output results file |

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
make CUDA=1 train-blt-d

bin-cuda/train_blt_d --corpus data/train.bin --eval-corpus data/heldout.bin \
    --eval-windows 300 --steps 40000 --diffusion 0 --lr-decay 1 --seed 7 \
    --save-weights runs/plain_40k.fblt --backend cuda

bin-cuda/train_blt_d --corpus data/train.bin --eval-corpus data/heldout.bin \
    --eval-windows 300 --steps 40000 --diffusion 1 --t-min 0.1 \
    --mask-warmup 10000 --mask-scale 0.3 --lr-decay 1 --seed 7 \
    --save-weights runs/bltd_l03_40k.fblt --backend cuda
```

### Toy-scale training with AdamW
```bash
bin-cuda/train_blt_d --corpus data/train.bin --eval-corpus data/heldout.bin \
    --eval-windows 300 --steps 40000 --diffusion 1 --t-min 0.1 \
    --mask-warmup 10000 --mask-scale 0.3 --lr-decay 1 --seed 7 \
    --optimizer adamw --lr 0.001 --weight-decay 0.01 \
    --save-weights runs/bltd_adamw_40k.fblt --backend cuda
```

### Inference comparison
```bash
make CUDA=1 bench-infer

bin-cuda/infer_bench --plain runs/plain_40k.fblt --bltd runs/bltd_l03_40k.fblt \
    --backend cuda --new-bytes 64 --results bench/results.jsonl
```

### At-scale sweep (27 runs)
```bash
tools/run_at_scale_sweep.sh --backend cuda
```
