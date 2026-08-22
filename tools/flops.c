#include "flops.h"

/* ---------- F1: Per-operation FLOPs primitives (Table 11) ---------- */

double blt_flops_attention(size_t l, size_t h_k, size_t n_heads, size_t m) {
    /* 4 * l * h_k * n_heads * (m+1)/2 */
    return 4.0 * (double)l * (double)h_k * (double)n_heads * ((double)m + 1.0) / 2.0;
}

double blt_flops_qkvo(size_t l, size_t h, double r) {
    /* (r*2 + 2) * 2 * l * h^2 */
    return (r * 2.0 + 2.0) * 2.0 * (double)l * (double)h * (double)h;
}

double blt_flops_feedforward(size_t l, size_t h, size_t d_ff) {
    /* 2 * l * 2 * h * (d_ff * h)
     * d_ff is a multiplier here (paper: d_ff = 4), matching Table 11 literally. */
    double ffn_width = (double)d_ff * (double)h;
    return 2.0 * (double)l * 2.0 * (double)h * ffn_width;
}

double blt_flops_deembedding(size_t h, size_t vocab_size) {
    /* 2 * h * |V| */
    return 2.0 * (double)h * (double)vocab_size;
}

double blt_flops_cross_attention(size_t l, size_t h_k, size_t n_heads,
                                  size_t patch_size, double r) {
    /* Attention(l, h_k, n_heads, patch_size) + QKVO(l, h_k * n_heads, r) */
    double attn = blt_flops_attention(l, h_k, n_heads, patch_size);
    double qkvo = blt_flops_qkvo(l, h_k * n_heads, r);
    return attn + qkvo;
}

/* ---------- F2: Per-module transformer FLOPs (Eq. 19-22) ---------- */

double blt_flops_transformer_block(
    size_t l, size_t h, size_t m, size_t n_heads, size_t h_k,
    size_t d_ff, size_t vocab_size
) {
    double ff        = blt_flops_feedforward(l, h, d_ff);
    double qkvo      = blt_flops_qkvo(l, h, 1.0);          /* r = 1 */
    double attn      = blt_flops_attention(l, h_k, n_heads, m);
    double deembed   = blt_flops_deembedding(h, vocab_size); /* 0 if vocab_size == 0 */

    return ff + qkvo + attn + deembed;
}

/* ---------- F3: Full BLT FLOPs-per-byte (Eq. 13-17) ---------- */

double blt_flops_per_byte(const blt_flops_config* cfg) {
    if (cfg == NULL || cfg->n_p == 0 || cfg->k == 0) {
        return 0.0;
    }

    double n_p = (double)cfg->n_p;
    double k   = (double)cfg->k;

    /* Eq. 13: global latent transformer, run once per patch, amortized per byte */
    size_t global_ctx = cfg->n_ctx / cfg->n_p; /* m = n_ctx / n_p, in patches */
    double FL_global = blt_flops_transformer_block(
        cfg->l_G, cfg->h_G, global_ctx,
        cfg->global_heads, cfg->global_head_dim,
        cfg->d_ff_G,
        0 /* vocab_size = 0, global model has no de-embedding */
    ) / n_p;

    /* Eq. 14: local encoder */
    double FL_encoder = blt_flops_transformer_block(
        cfg->l_E, cfg->h_E, cfg->w_E,
        cfg->enc_heads, cfg->enc_head_dim,
        cfg->d_ff_E,
        0 /* vocab_size = 0, encoder has no de-embedding */
    );

    /* Eq. 15: local decoder */
    double FL_decoder = blt_flops_transformer_block(
        cfg->l_D, cfg->h_D, cfg->w_D,
        cfg->dec_heads, cfg->dec_head_dim,
        cfg->d_ff_D,
        cfg->vocab_size /* 256 */
    );

    /* Eq. 16: encoder cross-attention, scaled by k/n_p */
    double FL_cross_E = blt_flops_cross_attention(
        cfg->l_E, cfg->h_E, cfg->enc_heads,
        cfg->n_p, n_p / k
    ) * (k / n_p);

    /* Eq. 17: decoder cross-attention */
    double FL_cross_D = blt_flops_cross_attention(
        cfg->l_D, cfg->h_D, cfg->dec_heads,
        cfg->k, k / n_p
    );

    return FL_global + FL_encoder + FL_decoder + FL_cross_E + FL_cross_D;
}