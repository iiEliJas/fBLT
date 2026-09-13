# Benchmarks

Inference verification at 2.97M parameters (embed=192, hidden=384, 2/2/2 layers, 40k steps, 83% warmup, mask-scale 0.3, high-t warmup on).

## Headline finding

**At 2.97M params, all verified inference methods (BLT-S, BLT-DV) cost more memory bandwidth than greedy.** BLT-DV acceptance rates (1.6-5.2%) are 18-42x lower than the paper's 3B results. Root cause likely because of to the 340x scale gap (paper's smallest model is 1B params). **OPEN** item: follow-up testing on a larger checkpoint needed.

## Setup

| | |
|---|---|
| Checkpoint | `s101.fblt` (2.97M params) | (Note: public checkpoint will be uploaded soon)
| Corpus | 67.5 MB C code (`data/train.bin`), 3.7 MB held-out (`data/heldout.bin`) |
| Model | BLT: local encoder + global patch transformer + local decoder, 2 layers each, E=192, H=384 |
| Prompts | 8 held-out prompts, 64 bytes each so combined over 512 bytes per config |
| Quality metric | BPB (bits per byte, lower is better) |
| Speed metrics | NFE (forward passes per byte), bandwidth (Eq. 8 from Fast-BLT paper) |

## Inference methods

| Method | How it generates | Expected cost |
|---|---|---|
| **greedy** (baseline) | one full encoder+global+decoder forward per byte | 1.0 decoder NFE/byte, 1.0 enc NFE/byte |
| **BLT-S** (self-speculation) | draft k bytes with decoder-only passes, verify all k with one full forward and accept the longest matching prefix | more decoder NFEs, far fewer encoder NFEs |
| **BLT-D** (block diffusion) | start a block of B masked positions, iteratively unmask the most confident ones; accept drafts as-is | cheapest per byte, output may diverge from greedy |
| **BLT-DV** (diffusion + verification) | BLT-D drafts, then a full causal forward verifies them like BLT-S | output matches greedy; draft makes it cheaper |

## Results

### Bandwidth analysis (Eq. 8)

Memory bandwidth = `b * [N_dec * P_dec + N_enc * (P_enc + P_glob)] / 10^9` GB, where b=2 (fp16), P_dec=1,131,840, P_enc+P_glob=1,839,744. Greedy baseline = 5.94 MB.

#### Entropy patching (default)

| Method | dec/byte | enc/byte | BW (MB) | BW ratio vs greedy |
|---|---:|---:|---:|---:|
| bltd greedy | 1.000 | 1.000 | 5.94 | 1.000 |
| bltd selfspec k=4 | 2.221 | 0.904 | 8.35 | 1.406 |
| bltd selfspec k=8 | 3.883 | 0.904 | 12.12 | 2.039 |
| bltd selfspec k=16 | 6.855 | 0.904 | 18.84 | 3.171 |
| bltd blockdiff B=4 | 0.441 | 0.250 | 1.92 | 0.323 |
| bltd blockdiff B=8 | 0.311 | 0.125 | 1.16 | 0.196 |
| bltd blockdiff B=16 | 0.225 | 0.062 | 0.74 | 0.124 |
| bltd blockdv B=4 | 1.994 | 1.672 | 10.67 | 1.795 |
| bltd blockdv B=8 | 2.318 | 1.660 | 11.36 | 1.911 |
| bltd blockdv B=16 | 2.852 | 1.645 | 12.51 | 2.105 |
| bltd blockdv onestep | 1.656 | 1.656 | 9.84 | 1.656 |
| bltd blockdv_eb g=1.0 | 2.480 | 1.660 | 11.72 | 1.972 |
| bltd blockdv_eb g=2.0 | 2.035 | 1.660 | 10.71 | 1.803 |
| bltd blockdiff_eb | 0.396 | 0.125 | 1.36 | 0.228 |
| bltd blockdv_hetv | 2.318 | 1.660 | 11.36 | 1.911 |
| bltd blockdv_bal | 2.709 | 1.926 | 13.22 | 2.224 |
| bltd blockdv_adapt | 2.014 | 1.672 | 10.71 | 1.802 |
| bltd blockdv_adapt_bal | 2.332 | 1.930 | 12.38 | 2.083 |

Raw BLT-D (blockdiff) is cheaper than greedy (0.12-0.32x), but output diverges (agreement 0.05-0.06). BLT-S and BLT-DV both cost more than greedy at this scale (1.41-3.17x).

#### Fixed-stride-4 patching

| Method | dec/byte | enc/byte | agree | BW (MB) | BW ratio vs greedy |
|---|---:|---:|---:|---:|---:|
| bltd greedy | 1.000 | 1.000 | 1.000 | 5.94 | 1.000 |
| bltd selfspec k=4 | 2.102 | 0.859 | 0.203 | 7.92 | 1.332 |
| bltd selfspec k=8 | 3.645 | 0.857 | 0.199 | 11.40 | 1.919 |
| bltd selfspec k=16 | 6.402 | 0.857 | 0.199 | 17.65 | 2.969 |
| bltd blockdiff B=4 | 0.424 | 0.250 | 0.055 | 1.88 | 0.316 |
| bltd blockdiff B=8 | 0.291 | 0.125 | 0.049 | 1.12 | 0.188 |
| bltd blockdiff B=16 | 0.254 | 0.062 | 0.045 | 0.80 | 0.135 |
| bltd blockdv B=4 | 2.312 | 1.789 | 0.469 | 11.82 | 1.988 |
| bltd blockdv B=8 | 2.725 | 1.762 | 0.463 | 12.65 | 2.129 |
| bltd blockdv B=16 | 3.518 | 1.758 | 0.463 | 14.43 | 2.428 |
| bltd blockdv onestep | 1.762 | 1.762 | 0.463 | 10.47 | 1.762 |
| bltd blockdv_eb g=1.0 | 3.018 | 1.762 | 0.463 | 13.32 | 2.240 |
| bltd blockdv_eb g=2.0 | 2.455 | 1.762 | 0.463 | 12.04 | 2.026 |
| bltd blockdiff_eb | 0.312 | 0.125 | 0.051 | 1.17 | 0.196 |
| bltd blockdv_hetv | 2.725 | 1.762 | 0.463 | 12.65 | 2.129 |

![Quality/cost frontier of generation methods](../graphs/nfe_quality_frontier.png)
*Decoder NFE/byte vs. agreement with greedy output, entropy patching, 2.97M-param checkpoint.*

![Where each method spends its forward passes](../graphs/enc_dec_map.png)
*Encoder+global NFE/byte vs. decoder NFE/byte, entropy patching, 2.97M-param checkpoint. BLT-D variants cluster near the origin; BLT-S variants are all over the x-axis as k increases.*

### Acceptance rates vs paper

The Fast-BLT paper reports acceptance rates at 1B-3B params on FR→EN translation. This repro is 2.97M params on C source code (~340x smaller). Different domain, different scale so expect different numbers. But we will still compare them:

**BLT-S acceptance rates (C code, 2.97M params vs paper's 3B FR→EN):**

| k | Paper 3B (FR→EN) | This repro 2.97M (C code) | Gap | Confidence |
|---|---:|---:|---|---|
| 4 | 94.9% | 30.8% | 3.1x lower | CONFIRMED |
| 8 | 87.2% | 15.8% | 5.5x lower | CONFIRMED |
| 16 | 69.9% | 8.5% | 8.2x lower | CONFIRMED |

**BLT-DV acceptance rates (C code, 2.97M params vs paper's 3B FR→EN):**

| Config | Paper 3B (FR→EN) | This repro 2.97M (C code) | Gap | Confidence |
|---|---:|---:|---|---|
| B=4, α=0.3 | 94.4% | 5.2% | 18.1x lower | CONFIRMED |
| B=8, α=0.3 | 86.3% | 2.8% | 30.8x lower | CONFIRMED |
| B=16, α=0.3 | 67.2% | 1.6% | 42.0x lower | CONFIRMED |
| B=8, one-step | 84.6% | 2.9% | 29.2x lower | CONFIRMED |

![Speculative acceptance rates by method](../graphs/acceptance.png)
*Drafted-byte acceptance rate by method, 2.97M-param checkpoint. BLT-S k=4 leads at 31%, declining with k. BLT-DV variants cluster at 2–25%.*

Note: `blockdv_onestep` config uses B=8 (per `BLTD_CFGS_ALL` in `infer_bench.c`), so the paper comparison uses the paper's BLT-D-8 3B one-step row (84.63%), not B=4's 93.12%. Paper results are FR→EN; this repro is C source code (the-stack-smol).

### Wall-clock latency

![Mean wall-clock latency per prompt](../graphs/latency.png)
*Mean wall-clock ms per prompt (64 new bytes). KV-cache usage differs between inference paths, so these are rough wall-clock numbers, not a clean comparison.*

### Entropy vs fixed-stride patching

| Patching | agree (BLT-DV) | BW ratio |
|---|---:|---:|---|
| Entropy | 1.000 | 1.66-2.11x |
| Fixed-stride-4 | 0.463-0.469 | 1.76-2.43x |

At this checkpoint, entropy patching gives byte-identical output between BLT-DV and greedy (agree=1.0) for all BLT-DV variants. Fixed-stride-4 breaks it so much that only ~46% of bytes match greedy. Fixed-stride patching doesn't work here because commit points land mid-patch.

## Reproducing

```bash
make CUDA=1 train-blt-d
./bin-cuda/train_blt_d --backend cuda \
  --corpus data/train.bin --eval-corpus data/heldout.bin \
  --embed 192 --hidden 384 --layers 2 \
  --steps 40000 --lr 0.05 --lr-decay 1 \
  --mask-warmup 33200 --mask-scale 0.3 \
  --t-min 0.1 --t-warmup-hi 0.25 --t-hi-start 0.8 \
  --cross-attn all --optimizer sgd \
  --diffusion 1 --block-size 4 --window 48 \
  --save-weights runs/my_checkpoint.fblt

make CUDA=1 bench-infer
./bin-cuda/infer_bench --backend cuda \
  --plain runs/my_checkpoint.fblt \
  --bltd runs/my_checkpoint.fblt \
  --prompts 8 --new-bytes 64
```

## CUDA benchmarks

All numbers from `bench/cuda_bench` on a **desktop NVIDIA GeForce RTX 4060** (AD107, 3072 fp32 cores, 115 W, 8188 MiB GDDR6, driver 616.56). CPU baseline is **single-threaded naive loops** (no OpenMP, no BLAS so really weak).

### Hardware identification

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 (AD107, CC 8.9) |
| VRAM | 8188 MiB GDDR6 |
| Power | 115 W |
| Max SM clock | 3105 MHz |
| fp32 peak (NVIDIA spec) | 15.11 TFLOP/s (3072 cores × 2460 MHz boost × 2) |
| Harness constant (corrected) | 15.11 TFLOP/s (`cuda_bench.c`) |

### Micro benchmarks (bench config: E=64, H=128, L=2)

Pure-op throughput, 3 runs averaged.

| Op | Size | CUDA (mean) | Unit |
|---|---|---|---|---|
| fp32 matmul | 512×512 | 5.87 | TFLOP/s |
| fp32 matmul | 1024×1024 | 6.77 | TFLOP/s | |
| fp32 matmul | 2048×2048 | 6.57 | TFLOP/s | |
| bf16 matmul | 512×512 | 11.2 | TFLOP/s |
| bf16 matmul | 1024×1024 | 18.73 | TFLOP/s | |
| bf16 matmul | 2048×2048 | 20.48 | TFLOP/s | |

fp32 matmul range: 5.87–6.77 TFLOP/s across sizes. bf16/fp32 ratio ranges from 1.9x (512) to 3.1x (2048).

### Meso benchmarks (bench config: E=64, L=2, full forward+backward pipeline)

All times in milliseconds (ms), 3 runs averaged.

| Config | Size | fwd (ms) | bwd (ms) |
|---|---|---|---|
| fp32 | 256 tokens | 4.28 | 15.76 |
| fp32 | 1024 tokens | 13.75 | 62.07 |
| bf16 | 256 tokens | 4.13 | 15.65 |
| bf16 | 1024 tokens | 13.68 | 62.11 |

![CPU vs CUDA fp32 vs CUDA bf16 pipeline timing](../graphs/cuda_speedup.png)
*CPU vs CUDA fp32 vs CUDA bf16 pipeline timing (E=64, H=128, L=2, fixed stride-4 patching). Seq=256: ~10x speedup. Seq=1024: ~24x speedup. Values averaged across multiple benchmark runs.*

### Macro benchmarks, training cell and generation cell

`macro_training_cell`: embed + 2×matmul + layernorm + SwiGLU + add (one full decoder layer, fwd+bwd). `macro_generation_cell`: autoregressive byte generation with KV-cache (naive greedy, 48-byte window). Both measured with `--iters 10`, 3 runs.

#### Bench config (E=64, H=128)

| Metric | CUDA (mean) | CPU (mean) | Speedup |
|---|---|---|---|
| macro_train_1024 (tokens/s) | 14,872 | 496 | **30.0x** |
| macro_gen_greedy (passes/s) | 221.7 | 6.3 | **35.2x** |

#### Production config (E=192, H=384, 2.97M params)

| Metric | CUDA (mean) | CPU (mean) | Speedup |
|---|---|---|---|
| macro_train_1024 (tokens/s) | 7,024 | 93 | **75.8x** |
| macro_gen_greedy (passes/s) | 217.7 | 6.3 | **34.4x** |

Per-run raw data (production config):

| Run | CUDA train | CUDA gen | CPU train | CPU gen |
|---|---:|---:|---:|---:|
| 1 | 7,004 | 216.2 | 92 | 6.3 |
| 2 | 7,005 | 222.4 | 93 | 6.3 |
| 3 | 7,062 | 214.5 | 93 | 6.4 |

Training speedup scales with model size (30x → 76x from E=64 to E=192) because the larger model better saturates GPU cores. Generation speedup stays flat (~34-35x).

### MFU analysis

| Metric | Value | Against |
|---|---|---|
| fp32 matmul mean (512/1024/2048) | 6.40 TFLOP/s | |
| **fp32 MFU** | **42.4%** | Desktop RTX 4060 spec peak (15.11 TFLOP/s) |
| bf16 matmul (1024+2048 avg) | 19.61 TFLOP/s | Tensor-core path |
| bf16/fp32 ratio at 512 | 1.9x | |
| bf16/fp32 ratio at 1024 | 2.8x | |
| bf16/fp32 ratio at 2048 | 3.1x | |
