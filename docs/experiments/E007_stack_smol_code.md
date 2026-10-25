# BLT-D on the-stack-smol C Code

## Metadata

- **Experiment ID:** E007
- **Title:** BLT-D on the-stack-smol C code
- **Status:** Planned
- **Date or reporting period:** Not started.
- **Author or contributors:** Not recorded yet.
- **Related reports, issues, or papers:** Byte Latent Transformer; Fast Byte Latent Transformer.
- **Git commit or release:** Not recorded yet.
- **Configurations and scripts:** `configs/train/stack_smol_c.yaml`
- **Artifacts:** Not created yet.

## 1. Research question

How well do BLT models work on source code rather than natural-language text?

## 2. Motivation and hypothesis

BLT's byte-level patching and diffusion decoding may transfer to source code, where
structure, indentation, and syntax differ from prose. A ~3M-parameter model on
~65 MB of C code is large enough to show whether the implementation trains
stably and produces plausible code completions.

## 3. Experimental design

- Dataset: `data/the-stack-smol` (`train.bin`, `heldout.bin`).
- Model: ~2.97M parameters, BLT-D, block size 4, 1024-byte window.
- Loss: paper-exact loss.
- Optimization: AdamW, cosine learning-rate schedule with 2000-step warmup.
- Patching: entropy patching using a code entropy LM.
- Steps: 100,000.
- Seed: 7.
- Evaluation: held-out causal BPB every 10,000 steps.

## 4. Reproduction instructions

Train the main model:

Train the code entropy LM first:

```bash
fblt-train --backend cuda \
  --corpus data/the-stack-smol/train.bin --window 1024 --steps 5000 --lr 0.1 \
  --train-entropy-lm runs/stack-smol-c/entropy_lm.fblt
```

Then train the main model:

```bash
fblt-train --config configs/train/stack_smol_c.yaml --backend cuda
```

## 5. Results

Not available yet.

## 6. Interpretation

Not available yet.

## 7. Limitations and validity threats

- Single dataset and model scale.
- The entropy LM is small and may not match the main model's segmentation needs.
- 100,000 steps may be insufficient for strong code generation.
- The held-out split may not represent all C coding styles.

## 8. Conclusion

Not available yet.

## 9. Follow-up work

Compare fixed-stride and entropy patching. Evaluate BLT-S, BLT-D, and BLT-DV on code.
Scale to larger models and datasets if the small run is stable.
