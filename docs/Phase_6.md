Fast-BLT Implementation Plan (BLT-S + BLT-D/DV)
What the code exploration showed (constraints that shape everything)
#	Fact	Consequence
1	D_0 = encoder h_final (local_decoder.c:293), not byte embeddings as in the paper	Drafted/MASKed rows have no decoder input → needs an explicit D_0 policy (your pick: zeros default + learned mode, config-switchable)
2	blt_model_forward fuses encoder→global→decoder (model.c:76)	Must stage-split so inference can freeze latents while the decoder loops
3	Decoder always computes shifted-CE loss tied to bytes_in	Inference wants a logits-only path (loss_out == NULL)
4	Mask builder supports causal/window/doc/groups (mask_builder_cpu.c:54) but masks can't depend on query row type	BLT-D masks need a dedicated new op, not struct overload
5	RoPE applied by row index == position (local_decoder.c:113)	Draft rows and BLT-D block rows sit at non-contiguous/original positions → need a position-gather util
6	Patcher is prefix-stable (patcher.c:98)	Completed patches are immutable → later KV-cache tiers are append-only; only the last patch ever rolls back
7	Only argmax_byte sampling exists (generate_greedy.c:16); byte_embedding.h already anticipates vocab 257 w/ MASK	EB sampling needs a top-p sampler; MASK token plumbing anticipated but unused

Phase A — Restructuring (zero behavior change, everything else builds on it)
A1. Stage-split the model forward (include/blt/models/model.h, src/models/model.c)
- blt_model_encode(...) → {patch_out P, global_out O, byte_hidden_out h_final}
- blt_model_decode(...) → decoder-only, accepts explicit D_0 source, nullable loss
- blt_model_forward becomes a wrapper calling both → existing parity/golden tests stay green untouched.
A2. Decoder extension (local_decoder.h/.c): new blt_local_decoder_forward_ext(...) taking an options struct: {loss_out nullable, d0_mode enum {HFINAL, ZEROS, LEARNED}, d0_extra_rows}. Old signature kept as wrapper. BLT_D0_LEARNED adds a [257, E] table + mask vector to decoder storage (flagged weight-layout change; trainer touch).
A3. Shared utils: rope_position_gather (rows for arbitrary position ids — used by both draft rows and BLT-D blocks); blt_infer_stats {nfe_dec, nfe_enc_glob, drafted, accepted} counted at controller level.
A4. Makefile: new include/blt/infer/ + src/infer/; sources appended to CORE_SRCS (tests link CORE_OBJS automatically).
Phase B — BLT-S (Algorithm 2, no retraining) — your chosen first target
Files: include/blt/infer/self_speculation.h, src/infer/self_speculation.c; config knob window_k ∈ {4,8,16}, d0_mode.
Round structure (per your choice: dense re-decode drafts, cache later):
1. Entropy LM → segment patches over committed prefix (patcher reused as-is)
2. One encoder+global call → frozen O (the expensive tier, 1 NFE)
3. Draft: up to k decoder-only passes; draft rows get cross-attn group id M−1 (condition on last latent, expressible with today's group mechanism) + gathered RoPE positions; D_0 rows per config
4. Verify (near-literal Algorithm 2): re-segment extended sequence, full blt_model_encode+decode → greedy predictions; accept until first mismatch, replace it, resume. Progress ≥ 1 byte/round guaranteed
5. Edge cases: k > remaining budget, mismatch at first draft byte, doc-boundary crossings
Tests (hard gates):
- Output-equivalence vs blt_generate_greedy on ≥50 prompts, byte-identical — silent off-by-ones live here
- Forced-mismatch injection at known position j → output matches plain AR through j, resumes at j+1
- Draft-conditioning mask unit test; NFE stats sanity (enc/global NFEs strictly < plain AR)
Phase C — Two-tier KV cache (after BLT-S is green, per your ordering)
include/blt/infer/kv_cache.h + src/infer/kv_cache.c: tier-1 byte-window (decoder self-attn, rollback = truncate), tier-2 patch-subtoken cross-attn K/V (append-only thanks to fact #6). Cache-aware attention variants alongside dense ones — dense paths untouched for parity. Optional tier-3: global-transformer patch KV (global dominates cost; defer until measured). Test: cached vs full-recompute logits equality (not just sampled bytes).
Phase D — BLT-D training (full scope, confirmed)
1. Mask op first: blt_build_block_diffusion_mask (train mode: Fig. 5 grid; infer mode: §3.1.1 single-live-block). Fixture test with the Figure-5 matrix hardcoded before any training code
2. Preprocessing: block construction from patches (start at s_i, length B, PAD-pad, original position ids), t~U(0,1) corruption (seeded RNG)
3. Decoder dual-segment forward: clean rows causal (unchanged path, bit-identical when diffusion off); block rows bidirectional-in-block + causal-to-prefix, cross-attn to o_{i−1}, RoPE gather by original positions, block D_0 per config. Loss L_clean + L_mask/t
4. Backward — heaviest item: extend blt_local_decoder_backward with the diffusion branch (two loss terms, grads through gather/cross/self-attn into shared weights + grad_patch_in). Validate with numeric-gradient check before wiring the trainer
5. Trainer + overfit test: sgd-step any new params; tiny BLT-D overfit: causal-mode BPB within noise of plain tiny BLT (your Table 2 analog) — this catches broken joint training, not "diffusion tax"
Phase E — BLT-D inference + BLT-DV
Algorithm 1 loop (src/infer/block_diffusion.c): [MASK]^B block → iterate unmasking: confidence (α, force-progress fallback) and entropy-bounded (γ cumulative-entropy prefix + new blt_sample_topp util). do_verify=true reuses Phase B's Verify() verbatim → that is BLT-DV. Tests: DV with all-reject drafts ≡ AR path; one-step mode; NFE accounting.
Phase F — Benchmarking (Plan.md's metric set)
New bench/bench_phase6.c (Makefile target) over a fixed held-out code-completion prompt file; one results.jsonl row per (variant, setting) with nfe_dec, nfe_enc_glob, mem_bw_gb (Eq. 8 — needs a small blt_model_param_counts() helper), accept_rate, wall-clock via existing bench_collect_samples, BPB. bench_report.py needs no mandatory changes — its phase filter, --baseline, and --x/--y scatter already produce the Table-1-shaped report and the Pareto plot.