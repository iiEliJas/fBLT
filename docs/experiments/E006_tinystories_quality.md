# TinyStories Quality Over Training

## Metadata

- **Experiment ID:** E006
- **Title:** TinyStories quality over training
- **Status:** Checked
- **Date or reporting period:** Not recorded in the historical report.
- **Author or contributors:** Not recorded in the historical report.
- **Related reports, issues, or papers:** Byte Latent Transformer; Fast Byte Latent Transformer.
- **Git commit or release:** Not recorded in the historical report.
- **Configurations and scripts:** 7.8M-parameter BLT-D model on TinyStories with entropy patching, block size 4, and a 512-byte window. Exact training commands and seeds are not recorded here; evaluation uses held-out causal BPB.
- **Artifacts:** Held-out BPB measurements are recorded below. The `graphs/quality_vs_steps.png` figure is generated from those measurements.

## 1. Research question

How does held-out causal bits per byte improve as a 7.8M-parameter BLT-D model trains on
TinyStories?

## 2. Motivation and hypothesis

The model was trained to test whether the fBLT implementation produces a usable byte-level
language model at small scale. The expectation was that held-out BPB would decrease steadily
without overfitting.

## 3. Experimental design

- Model: 7.8M parameters, BLT-D, entropy patching, block size 4, window 512.
- Dataset: TinyStories.
- Metric: held-out causal bits per byte.
- Checkpoints evaluated: 30k, 360k, and 600k steps.
- Evaluation uses held-out bytes and does not overlap with training data.

## 4. Reproduction instructions

Exact training commands and seeds are not recorded in this historical report. The evaluation
command is:

```bash
./build-cuda/train_blt_d --corpus data/tinystories/train.bin \
  --load-weights runs/tinystories_600k/tinystories_600k.fblt --steps 0 \
  --embed 256 --hidden 512 --enc-layers 2 --glob-layers 6 --dec-layers 2 \
  --entropy-lm runs/entropylm/entropy_lm.fblt --entropy-patches \
  --eval-corpus data/tinystories/heldout.bin --eval-windows 200 --backend cuda
```

For the full workflow, see [STARTER_GUIDE.md](../STARTER_GUIDE.md).

## 5. Results

| Checkpoint | Training steps | Held-out causal BPB |
|---|---:|---:|
| 30k probe | 30,000 | 1.7536 |
| 360k | 360,000 | 1.1611 |
| 600k | 600,000 | **0.9978** |

![Held-out causal BPB vs training steps](../../graphs/quality_vs_steps.png)

## 6. Interpretation

Held-out causal BPB improves monotonically from 1.7536 to 0.9978 over 600k steps. There is
no observed overfitting on the held-out TinyStories split.

## 7. Limitations and validity threats

- Only one dataset and one model scale are measured.
- The model is 7.8M parameters, so results may not transfer to larger models.
- Training and evaluation use a fixed TinyStories split.
- Exact training commands and seeds are not recorded here.

## 8. Conclusion

The implementation trains a usable small byte-level language model on TinyStories, with steady
quality improvement and no observed overfitting.

## 9. Follow-up work

Run the same evaluation on additional datasets and at larger scales. Record full training
commands and seeds for future checkpoints.
