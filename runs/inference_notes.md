# Inference notes (BLT-D generation + BLT-DV)

Date: 2026-08-25. Builds on the trained checkpoints (see training_notes.md).

## What was built

- `blt_local_decoder_forward_diffusion_infer`: decoder pass with the
  paper's section 3.1.1 INFERENCE attention pattern (clean rows causal;
  block rows bidirectional over clean + whole live block; cross-attn to
  the last latent o_M). No loss path.
- `include/blt/infer/block_generation.h` + src/infer/block_generation.c:
  - selection kernels: confidence-based (alpha, best-cell fallback) and
    entropy-bounded (gamma budget over ascending entropies,
    lowest-entropy fallback); optional top-p prediction sampling.
  - `blt_draft_block`: Algorithm 1 inner loop against frozen latents.
  - `blt_generate_greedy_blockdiff`: Algorithm 1 without verification.
  - `blt_generate_greedy_blockdiff_verify`: BLT-DV -- drafts verified via
    the existing `blt_verify_draft` (Algorithm 2) reused verbatim.

## Correctness gates (all passing, 60/60 suite)

1. Selection kernels pinned on hand-computed score grids.
2. Draft behavior: determinism, byte validity, NFE accounting per
   strategy extreme (alpha-unreachable => B passes; alpha=0 or huge gamma
   => 1 pass), seeded top-p reproducibility.
3. **DV == plain greedy, byte for byte** -- unit test on a snippet-trained
   toy model AND `make e2e-dv` on the real runs/bltd_l03_40k.fblt
   checkpoint against a deep held-out prompt. This holds by construction
   (accept-until-mismatch + replacement + free byte), so it is the
   load-bearing regression gate.
4. Exact output lengths, per-round progress, stats accounting.

## Toy-scale findings

- Raw drafts are code-shaped but diverge from causal argmaxes: acceptance
  ~14% on the real checkpoint (B=8, alpha=0.7). Decoder NFEs therefore do
  not beat one-per-byte at this scale yet.
- Root cause is consistent with training-day results: the fully-masked
  (t~1) end of the corruption distribution is under-trained when L_mask
  is capped (lambda<=0.3) and data/steps are small; large-scale models
  (paper section 5.2) get high acceptance exactly there.
- One real bug found and fixed during bring-up: blt_draft_block reset the
  scratch arena per pass, freeing the frozen encoder latents it was
  handed (uniform-logit symptom). Arena contract documented in the header.

## Try it

    make e2e-dv                       # default prompt/checkpoint
    ./bin/e2e_blt_dv --block-size 4 --new-bytes 64 \
        --prompt "int main(void) {"

## Next

Benchmark harness comparing BLT / BLT-S / BLT-D / BLT-DV NFEs and
acceptance across k and B sweeps on held-out text; optional longer BLT-D
runs with higher lambda cap late in training to lift t~1 quality.
