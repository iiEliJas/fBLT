#include "test_helpers.h"
#include "test_suite.h"

#include "models/entropy_lm.h"
#include "core/allocator.h"
#include "core/tensor.h"

#include <math.h>
#include <stdint.h>

// Deterministic, bounded pseudo-random fill so gradients are non-trivial
// (all-zero weights would make most of this graph trivially zero).
static void fill_deterministic(blt_tensor *t, float seed) {
    float *d = (float *)t->data;
    for (size_t i = 0; i < t->numel; ++i) {
        d[i] = 0.1f * sinf((float)i * 0.37f + seed);
    }
}

static void fill_model(blt_entropy_lm *model) {
    fill_deterministic(&model->embedding_weight, 0.0f);
    fill_deterministic(&model->lm_head_weight, 1.0f);
    for (size_t l = 0; l < model->config.num_layers; ++l) {
        blt_transformer_layer_storage *s = &model->stack.layer_storage[l];
        fill_deterministic(&s->norm1_weight, 2.0f + (float)l);
        fill_deterministic(&s->norm2_weight, 3.0f + (float)l);
        fill_deterministic(&s->attn_qkv_w, 4.0f + (float)l);
        fill_deterministic(&s->attn_proj_w, 5.0f + (float)l);
        fill_deterministic(&s->ffn_up_w, 6.0f + (float)l);
        fill_deterministic(&s->ffn_gate_w, 7.0f + (float)l);
        fill_deterministic(&s->ffn_down_w, 8.0f + (float)l);
        // RMSNorm weights start near 1.0 so norm doesnt zero everything out
        float *n1 = (float *)s->norm1_weight.data;
        float *n2 = (float *)s->norm2_weight.data;
        for (size_t i = 0; i < s->norm1_weight.numel; ++i) n1[i] += 1.0f;
        for (size_t i = 0; i < s->norm2_weight.numel; ++i) n2[i] += 1.0f;
    }
}

static blt_entropy_lm_config make_small_config(void) {
    blt_entropy_lm_config cfg;
    cfg.embed_dim = 4;
    cfg.num_layers = 1;
    cfg.hidden_dim = 8;
    cfg.num_heads = 2;
    cfg.max_seq_len = 8;
    cfg.rope_theta = 10000.0f;
    return cfg;
}

// ---------------------------------------------------------------------
// Shape/wiring smoke test

static int test_entropy_lm_forward_smoke(void) {
    blt_arena *model_arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *scratch = blt_arena_create(4 * 1024 * 1024, BLT_BACKEND_CPU);
    if (!model_arena || !scratch) {
        return 0;
    }

    blt_entropy_lm_config cfg = make_small_config();
    blt_entropy_lm *model = blt_entropy_lm_create(model_arena, &cfg);
    fill_model(model);

    size_t seq_len = 5;
    size_t bytes_shape[1] = {seq_len};
    blt_tensor bytes_in = blt_tensor_create(model_arena, bytes_shape, 1, BLT_DTYPE_UINT8);
    uint8_t *b = (uint8_t *)bytes_in.data;
    uint8_t vals[5] = {'h', 'e', 'l', 'l', 'o'};
    for (size_t i = 0; i < seq_len; ++i) b[i] = vals[i];

    size_t logits_shape[2] = {seq_len, 256};
    blt_tensor logits = blt_tensor_create(scratch, logits_shape, 2, BLT_DTYPE_FP32);
    size_t loss_shape[1] = {1};
    blt_tensor loss = blt_tensor_create(scratch, loss_shape, 1, BLT_DTYPE_FP32);

    blt_entropy_lm_forward(model, &bytes_in, &logits, &loss, scratch);

    int ok = 1;

    // Shape check.
    ok &= (logits.ndim == 2 && logits.shape[0] == seq_len && logits.shape[1] == 256);
    TEST_ASSERT(ok);

    // No-NaN check across all logits and the loss.
    float *ld = (float *)logits.data;
    for (size_t i = 0; i < logits.numel; ++i) {
        if (isnan(ld[i]) || isinf(ld[i])) {
            ok = 0;
            break;
        }
    }
    TEST_ASSERT(ok);

    float loss_val = ((float *)loss.data)[0];
    ok &= !(isnan(loss_val) || isinf(loss_val));
    ok &= (loss_val > 0.0f); // cross-entropy over a non-degenerate distribution is positive
    TEST_ASSERT(ok);

    blt_arena_destroy(scratch);
    blt_arena_destroy(model_arena);
    return ok;
}

// ---------------------------------------------------------------------
// End-to-end finite-difference gradient check

// Runs a fresh forward pass (using a reset scratch arena) and returns the
// scalar loss. Used both for the "real" loss and for perturbed losses.
static float forward_loss(const blt_entropy_lm *model, const blt_tensor *bytes_in, size_t seq_len, blt_arena *scratch) {
    blt_arena_reset(scratch);
    size_t logits_shape[2] = {seq_len, 256};
    blt_tensor logits = blt_tensor_create(scratch, logits_shape, 2, BLT_DTYPE_FP32);
    size_t loss_shape[1] = {1};
    blt_tensor loss = blt_tensor_create(scratch, loss_shape, 1, BLT_DTYPE_FP32);
    blt_entropy_lm_forward(model, bytes_in, &logits, &loss, scratch);
    return ((float *)loss.data)[0];
}

static int test_entropy_lm_finite_difference_gradient(void) {
    blt_arena *model_arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *scratch = blt_arena_create(8 * 1024 * 1024, BLT_BACKEND_CPU);
    if (!model_arena || !scratch) {
        return 0;
    }

    blt_entropy_lm_config cfg = make_small_config();
    blt_entropy_lm *model = blt_entropy_lm_create(model_arena, &cfg);
    fill_model(model);

    size_t seq_len = 4;
    size_t bytes_shape[1] = {seq_len};
    blt_tensor bytes_in = blt_tensor_create(model_arena, bytes_shape, 1, BLT_DTYPE_UINT8);
    uint8_t *b = (uint8_t *)bytes_in.data;
    b[0] = 'a';
    b[1] = 'b';
    b[2] = 'c';
    b[3] = 'd';

    // Analytic gradient (single backward call; scratch is reset immediately
    // after so it's free for the finite-difference forward passes below).
    blt_arena_reset(scratch);
    blt_entropy_lm_grad *grad = blt_entropy_lm_grad_create(scratch, model);
    blt_entropy_lm_backward(model, &bytes_in, grad, scratch);

    // Copy out the tensors we'll check against before further scratch use
    // (grad_create/backward both live in `scratch`, but we don't reset
    // scratch again until after the checks below, so this is safe as-is —
    // kept as separate tensors for clarity of what's being compared).
    blt_tensor embedding_grad = grad->embedding_grad;
    blt_tensor norm1_grad = grad->stack_grad->layer_grads[0].norm1_weight;
    blt_tensor ffn_down_grad = grad->stack_grad->layer_grads[0].ffn_down_w;

    // NOTE: forward_loss() below calls blt_arena_reset(scratch), which
    // would invalidate the grad tensors above. Snapshot the specific
    // elements we're going to check as plain floats first.
    float analytic_embed = ((float *)embedding_grad.data)[3]; // embedding_weight[0][3]
    float analytic_norm1 = ((float *)norm1_grad.data)[1];
    float analytic_ffn_down = ((float *)ffn_down_grad.data)[5];

    size_t embed_dim = model->config.embed_dim;
    (void)embed_dim;

    float eps = 1e-3f;
    int ok = 1;

    // -- embedding_weight[0][3] --
    {
        float *w = (float *)model->embedding_weight.data;
        size_t idx = 3;
        float original = w[idx];
        w[idx] = original + eps;
        float loss_plus = forward_loss(model, &bytes_in, seq_len, scratch);
        w[idx] = original - eps;
        float loss_minus = forward_loss(model, &bytes_in, seq_len, scratch);
        w[idx] = original;
        float numeric = (loss_plus - loss_minus) / (2.0f * eps);
        float diff = numeric - analytic_embed;
        if (diff < 0) diff = -diff;
        float tol = 5e-2f + 5e-2f * (numeric < 0 ? -numeric : numeric);
        ok &= (diff <= tol);
    }
    TEST_ASSERT(ok);

    // -- layer 0 norm1_weight[1] --
    {
        float *w = (float *)model->stack.layer_storage[0].norm1_weight.data;
        size_t idx = 1;
        float original = w[idx];
        w[idx] = original + eps;
        float loss_plus = forward_loss(model, &bytes_in, seq_len, scratch);
        w[idx] = original - eps;
        float loss_minus = forward_loss(model, &bytes_in, seq_len, scratch);
        w[idx] = original;
        float numeric = (loss_plus - loss_minus) / (2.0f * eps);
        float diff = numeric - analytic_norm1;
        if (diff < 0) diff = -diff;
        float tol = 5e-2f + 5e-2f * (numeric < 0 ? -numeric : numeric);
        ok &= (diff <= tol);
    }
    TEST_ASSERT(ok);

    // -- layer 0 ffn_down_w[5] --
    {
        float *w = (float *)model->stack.layer_storage[0].ffn_down_w.data;
        size_t idx = 5;
        float original = w[idx];
        w[idx] = original + eps;
        float loss_plus = forward_loss(model, &bytes_in, seq_len, scratch);
        w[idx] = original - eps;
        float loss_minus = forward_loss(model, &bytes_in, seq_len, scratch);
        w[idx] = original;
        float numeric = (loss_plus - loss_minus) / (2.0f * eps);
        float diff = numeric - analytic_ffn_down;
        if (diff < 0) diff = -diff;
        float tol = 5e-2f + 5e-2f * (numeric < 0 ? -numeric : numeric);
        ok &= (diff <= tol);
    }
    TEST_ASSERT(ok);

    blt_arena_destroy(scratch);
    blt_arena_destroy(model_arena);
    return ok;
}

int run_elm_model_tests(void) {
    int ok = 1;
    ok &= test_entropy_lm_forward_smoke();
    ok &= test_entropy_lm_finite_difference_gradient();
    return ok;
}