#include "test_helpers.h"
#include "test_suite.h"
#include "flops.h"

/* ---------------------------------------------------------------
 * F4.1 — Hand-derived example, using the 400M row of Table 10:
 *   Encoder: l=1,  heads=12, h=768,  head_dim=64
 *   Global:  l=24, heads=10, h=1280, head_dim=128
 *   Decoder: l=7,  heads=12, h=768,  head_dim=64
 *   Cross-attn heads=10, k=2, d_ff multiplier=4 throughout,
 *   n_ctx=8192, n_p=4, w_E=w_D=512, vocab_size=256.
 *
 * Hand-derived total (Eq. 13-17), computed independently:
 *   FL_global   = 267,402,240
 *   FL_encoder  =  14,943,744
 *   FL_decoder  = 104,999,424
 *   FL_cross_E  = 353,932,800
 *   FL_cross_D  = 2,477,583,360
 *   ----------------------------
 *   Total       = 3,218,861,568 FLOPs/byte
 * --------------------------------------------------------------- */
int run_flops_hand_derived_test(void) {
    blt_flops_config cfg = {0};

    /* Global transformer */
    cfg.h_G = 1280;
    cfg.l_G = 24;
    cfg.n_ctx = 8192;
    cfg.global_heads = 10;
    cfg.global_head_dim = 128;
    cfg.d_ff_G = 4;

    /* Local encoder */
    cfg.h_E = 768;
    cfg.l_E = 1;
    cfg.w_E = 512;
    cfg.enc_heads = 10;       /* cross-attn heads, per Table 10 */
    cfg.enc_head_dim = 64;
    cfg.d_ff_E = 4;

    /* Local decoder */
    cfg.h_D = 768;
    cfg.l_D = 7;
    cfg.w_D = 512;
    cfg.dec_heads = 10;       /* cross-attn heads, per Table 10 */
    cfg.dec_head_dim = 64;
    cfg.d_ff_D = 4;

    /* Shared */
    cfg.n_p = 4;
    cfg.k = 2;
    cfg.vocab_size = 256;

    double computed = blt_flops_per_byte(&cfg);
    double expected = 3218861568.0;

    /* exact integer arithmetic in double, so a tight relative
     * tolerance is appropriate here (not a numerical-noise check) */
    double rel_diff = (computed - expected) / expected;
    if (rel_diff < 0.0) rel_diff = -rel_diff;

    TEST_ASSERT(rel_diff < 1e-6);

    return 1;
}

/* ---------------------------------------------------------------
 * F4.2 — Scaling sanity check: doubling n_p (patch size) at fixed
 * everything else should reduce total FLOPs/byte, and specifically
 * should reduce it roughly proportionally (since FL_global is
 * divided by n_p directly, and the encoder/decoder cross-attention
 * terms also shrink as n_p grows). This is a cheap invariant test
 * that would catch a sign error or a missing division by n_p.
 * --------------------------------------------------------------- */
int run_flops_patch_size_scaling_test(void) {
    blt_flops_config cfg = {0};

    cfg.h_G = 1280;
    cfg.l_G = 24;
    cfg.n_ctx = 8192;
    cfg.global_heads = 10;
    cfg.global_head_dim = 128;
    cfg.d_ff_G = 4;

    cfg.h_E = 768;
    cfg.l_E = 1;
    cfg.w_E = 512;
    cfg.enc_heads = 10;
    cfg.enc_head_dim = 64;
    cfg.d_ff_E = 4;

    cfg.h_D = 768;
    cfg.l_D = 7;
    cfg.w_D = 512;
    cfg.dec_heads = 10;
    cfg.dec_head_dim = 64;
    cfg.d_ff_D = 4;

    cfg.k = 2;
    cfg.vocab_size = 256;

    cfg.n_p = 4;
    double flops_np4 = blt_flops_per_byte(&cfg);

    cfg.n_p = 8;
    double flops_np8 = blt_flops_per_byte(&cfg);

    /* Must strictly decrease */
    TEST_ASSERT(flops_np8 < flops_np4);

    /* Should be roughly proportional: expect the doubled-patch-size
     * total to land somewhere in [30%, 70%] of the original, a loose
     * band that still catches a broken/missing division by n_p
     * (which would show ~100% i.e. no change) or a sign flip
     * (which would show an increase). */
    double ratio = flops_np8 / flops_np4;
    TEST_ASSERT(ratio > 0.30 && ratio < 0.70);

    return 1;
}