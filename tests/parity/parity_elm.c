// Exit criteria being checked:
//   - Loss decreases monotone and reaches almost zero within a few hundred steps
//   - Passes on at least two different small batches (different pattern or
//     different weight init seed) to rule out a fluke of one input
//   - Loss visibly decreases on a small code heavy slice too

#include "test_helpers.h"
#include "test_suite.h"

#include "blt/models/entropy_lm.h"
#include "blt/ops/optim.h"
#include "blt/ops/elementwise.h"
#include "blt/core/allocator.h"
#include "blt/core/tensor.h"

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#define OVERFIT_BATCH_SIZE 8
#define OVERFIT_NUM_STEPS  400


//----------------------------------------------------------------------
// Helpers for init and fill

// Deterministic, bounded random fill
static void fill_deterministic(blt_tensor* t, float seed) {
    float* d = (float*)t->data;
    for (size_t i = 0; i < t->numel; ++i) {
        d[i] = 0.1f * sinf((float)i * 0.37f + seed);
    }
}

static void init_model_weights(blt_entropy_lm* model, float seed) {
    fill_deterministic(&model->embedding_weight, seed + 0.0f);
    fill_deterministic(&model->lm_head_weight, seed + 1.0f);
    for (size_t l = 0; l < model->config.num_layers; ++l) {
        blt_transformer_layer_storage* s = &model->layer_storage[l];
        fill_deterministic(&s->norm1_weight, seed + 2.0f + (float)l);
        fill_deterministic(&s->norm2_weight, seed + 3.0f + (float)l);
        fill_deterministic(&s->attn_qkv_w, seed + 4.0f + (float)l);
        fill_deterministic(&s->attn_proj_w, seed + 5.0f + (float)l);
        fill_deterministic(&s->ffn_up_w, seed + 6.0f + (float)l);
        fill_deterministic(&s->ffn_gate_w, seed + 7.0f + (float)l);
        fill_deterministic(&s->ffn_down_w, seed + 8.0f + (float)l);
        // RMSNorm weights start near 1.0 so normalization doesnt zero everything out
        float* n1 = (float*)s->norm1_weight.data;
        float* n2 = (float*)s->norm2_weight.data;
        for (size_t i = 0; i < s->norm1_weight.numel; ++i) n1[i] += 1.0f;
        for (size_t i = 0; i < s->norm2_weight.numel; ++i) n2[i] += 1.0f;
    }
}

// grad_acc += grad_sample, elementwise over every learnable weight
static void accumulate_grad(blt_entropy_lm_grad* acc, const blt_entropy_lm_grad* sample,
                             const blt_entropy_lm* model) {
    blt_add(&acc->embedding_grad, &sample->embedding_grad, &acc->embedding_grad);
    blt_add(&acc->lm_head_grad, &sample->lm_head_grad, &acc->lm_head_grad);
    for (size_t l = 0; l < model->config.num_layers; ++l) {
        blt_transformer_layer_grad* a = &acc->layer_grads[l];
        const blt_transformer_layer_grad* s = &sample->layer_grads[l];
        blt_add(&a->norm1_weight, &s->norm1_weight, &a->norm1_weight);
        blt_add(&a->attn_qkv_w, &s->attn_qkv_w, &a->attn_qkv_w);
        blt_add(&a->attn_proj_w, &s->attn_proj_w, &a->attn_proj_w);
        blt_add(&a->norm2_weight, &s->norm2_weight, &a->norm2_weight);
        blt_add(&a->ffn_up_w, &s->ffn_up_w, &a->ffn_up_w);
        blt_add(&a->ffn_gate_w, &s->ffn_gate_w, &a->ffn_gate_w);
        blt_add(&a->ffn_down_w, &s->ffn_down_w, &a->ffn_down_w);
    }
}

// Scales every learnable weight gradient tensor by s
static void scale_grad(blt_entropy_lm_grad* g, float s, const blt_entropy_lm* model) {
    blt_scale(&g->embedding_grad, s);
    blt_scale(&g->lm_head_grad, s);
    for (size_t l = 0; l < model->config.num_layers; ++l) {
        blt_transformer_layer_grad* lg = &g->layer_grads[l];
        blt_scale(&lg->norm1_weight, s);
        blt_scale(&lg->attn_qkv_w, s);
        blt_scale(&lg->attn_proj_w, s);
        blt_scale(&lg->norm2_weight, s);
        blt_scale(&lg->ffn_up_w, s);
        blt_scale(&lg->ffn_gate_w, s);
        blt_scale(&lg->ffn_down_w, s);
    }
}

// Applies one blt_sgd_step for every learnable weight tensor
static void apply_sgd_step(blt_entropy_lm* model, const blt_entropy_lm_grad* g, float lr) {
    blt_sgd_step(&model->embedding_weight, &g->embedding_grad, lr);
    blt_sgd_step(&model->lm_head_weight, &g->lm_head_grad, lr);
    for (size_t l = 0; l < model->config.num_layers; ++l) {
        blt_transformer_layer_storage* w = &model->layer_storage[l];
        const blt_transformer_layer_grad* lg = &g->layer_grads[l];
        blt_sgd_step(&w->norm1_weight, &lg->norm1_weight, lr);
        blt_sgd_step(&w->attn_qkv_w, &lg->attn_qkv_w, lr);
        blt_sgd_step(&w->attn_proj_w, &lg->attn_proj_w, lr);
        blt_sgd_step(&w->norm2_weight, &lg->norm2_weight, lr);
        blt_sgd_step(&w->ffn_up_w, &lg->ffn_up_w, lr);
        blt_sgd_step(&w->ffn_gate_w, &lg->ffn_gate_w, lr);
        blt_sgd_step(&w->ffn_down_w, &lg->ffn_down_w, lr);
    }
}

static float average_range(const float* losses, size_t start, size_t end) {
    float sum = 0.0f;
    size_t n = end - start;
    for (size_t i = start; i < end; ++i) sum += losses[i];
    return n > 0 ? sum / (float)n : 0.0f;
}


// Sum of squares of one tensors elements
static float sum_sq(const blt_tensor* t) {
    const float* d = (const float*)t->data;
    float s = 0.0f;
    for (size_t i = 0; i < t->numel; ++i) s += d[i] * d[i];
    return s;
}
 
static float grad_global_norm(const blt_entropy_lm_grad* g, const blt_entropy_lm* model) {
    float ss = sum_sq(&g->embedding_grad) + sum_sq(&g->lm_head_grad);
    for (size_t l = 0; l < model->config.num_layers; ++l) {
        const blt_transformer_layer_grad* lg = &g->layer_grads[l];
        ss += sum_sq(&lg->norm1_weight) + sum_sq(&lg->attn_qkv_w) + sum_sq(&lg->attn_proj_w)
            + sum_sq(&lg->norm2_weight) + sum_sq(&lg->ffn_up_w) + sum_sq(&lg->ffn_gate_w)
            + sum_sq(&lg->ffn_down_w);
    }
    return sqrtf(ss);
}
 
// Global-norm gradient clipping
static void clip_grad_norm(blt_entropy_lm_grad* g, const blt_entropy_lm* model, float max_norm) {
    float norm = grad_global_norm(g, model);
    if (norm > max_norm && norm > 0.0f) {
        scale_grad(g, max_norm / norm, model);
    }
}



//----------------------------------------------------------------------
// training loop

// Runs OVERFIT_NUM_STEPS steps of plain SGD over a batch of OVERFIT_BATCH_SIZE copies of one short byte string
// logging the loss every step 
// Returns 1 if the loss decreases overall and reaches near-zero, 0 otherwise

static int run_overfit_batch(const uint8_t* pattern, size_t seq_len, float init_seed,
                              float lr, float near_zero_threshold, const char* label) {
    blt_arena* model_arena = blt_arena_create(2 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena* grad_arena  = blt_arena_create(2 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena* work_arena  = blt_arena_create(8 * 1024 * 1024, BLT_BACKEND_CPU);
    if (!model_arena || !grad_arena || !work_arena) {
        return 0;
    }

    blt_entropy_lm_config cfg;
    cfg.embed_dim = 16;
    cfg.num_layers = 2;
    cfg.hidden_dim = 32;
    cfg.num_heads = 2;
    cfg.max_seq_len = seq_len > 32 ? seq_len : 32;
    cfg.rope_theta = 10000.0f;

    blt_entropy_lm* model = blt_entropy_lm_create(model_arena, &cfg);
    init_model_weights(model, init_seed);

    // Batch of OVERFIT_BATCH_SIZE identical copies of pattern
    blt_tensor bytes_in[OVERFIT_BATCH_SIZE];
    size_t bytes_shape[1] = { seq_len };
    for (int i = 0; i < OVERFIT_BATCH_SIZE; ++i) {
        bytes_in[i] = blt_tensor_create(model_arena, bytes_shape, 1, BLT_DTYPE_UINT8);
        memcpy(bytes_in[i].data, pattern, seq_len);
    }

    float losses[OVERFIT_NUM_STEPS];

    for (int step = 0; step < OVERFIT_NUM_STEPS; ++step) {
        blt_arena_reset(grad_arena);
        blt_entropy_lm_grad* grad_acc = blt_entropy_lm_grad_create(grad_arena, model);

        float step_loss_sum = 0.0f;
        for (int s = 0; s < OVERFIT_BATCH_SIZE; ++s) {
            // Forward pass for the logged loss value
            blt_arena_reset(work_arena);
            size_t logits_shape[2] = { seq_len, 256 };
            blt_tensor logits = blt_tensor_create(work_arena, logits_shape, 2, BLT_DTYPE_FP32);
            size_t loss_shape[1] = { 1 };
            blt_tensor loss = blt_tensor_create(work_arena, loss_shape, 1, BLT_DTYPE_FP32);
            blt_entropy_lm_forward(model, &bytes_in[s], &logits, &loss, work_arena);
            step_loss_sum += ((float*)loss.data)[0];

            // Backward pass for this sample
            blt_arena_reset(work_arena);
            blt_entropy_lm_grad* grad_sample = blt_entropy_lm_grad_create(work_arena, model);
            blt_entropy_lm_backward(model, &bytes_in[s], grad_sample, work_arena);
            accumulate_grad(grad_acc, grad_sample, model);
        }

        losses[step] = step_loss_sum / (float)OVERFIT_BATCH_SIZE;
 
        if (isnan(losses[step]) || isinf(losses[step])) {
            fprintf(stderr, "[overfit:%s] diverged at step %d (loss=%f) — aborting run\n",
                    label, step, losses[step]);
            for (int rest = step + 1; rest < OVERFIT_NUM_STEPS; ++rest) losses[rest] = losses[step];
            blt_arena_destroy(work_arena);
            blt_arena_destroy(grad_arena);
            blt_arena_destroy(model_arena);
            TEST_ASSERT(0);
            return 0;
        }
 
        scale_grad(grad_acc, 1.0f / (float)OVERFIT_BATCH_SIZE, model);
        clip_grad_norm(grad_acc, model, 1.0f); //max_norm = 1.0f
        apply_sgd_step(model, grad_acc, lr);
    }


    //--------------------------
    // Exit criteria checks

    // Loss decreases monotonic: 
    // compare the average loss over the first and last 10% of steps
    size_t window = OVERFIT_NUM_STEPS / 10;
    if (window < 1) window = 1;
    float first_avg = average_range(losses, 0, window);
    float last_avg  = average_range(losses, OVERFIT_NUM_STEPS - window, OVERFIT_NUM_STEPS);

    int decreasing = last_avg < first_avg;
    int near_zero = losses[OVERFIT_NUM_STEPS - 1] <= near_zero_threshold;

    // bound how noisy the trend is allowed to be:
    int upticks = 0;
    for (int step = 1; step < OVERFIT_NUM_STEPS; ++step) {
        if (losses[step] > losses[step - 1]) upticks++;
    }
    int not_too_noisy = upticks <= (OVERFIT_NUM_STEPS * 4) / 10;

    int ok = decreasing && near_zero && not_too_noisy;

    fprintf(stderr,
        "   [overfit %s] first_avg=%.4f last_avg=%.4f final=%.4f upticks=%d/%d -> %s\n",
        label, first_avg, last_avg, losses[OVERFIT_NUM_STEPS - 1], upticks, OVERFIT_NUM_STEPS - 1,
        ok ? "PASS" : "FAIL");

    TEST_ASSERT(ok);

    blt_arena_destroy(work_arena);
    blt_arena_destroy(grad_arena);
    blt_arena_destroy(model_arena);
    return ok;
}



//----------------------------------------------------------------------
// Test entry points

// First batch: a short repeating pattern
static int test_overfit_batch_pattern_a(void) {
    const uint8_t pattern[] = "abcabc";
    return run_overfit_batch(pattern, sizeof(pattern) - 1, 0.0f, 0.1f, 0.05f, "pattern_a");
}

// Second batch: different pattern and seed
static int test_overfit_batch_pattern_b(void) {
    const uint8_t pattern[] = "xyzzy!";
    return run_overfit_batch(pattern, sizeof(pattern) - 1, 17.0f, 0.1f, 0.05f, "pattern_b");
}

// Third batch: a small code slice
static int test_overfit_batch_code_slice(void) {
    const uint8_t pattern[] = "int main(){return 0;}";
    return run_overfit_batch(pattern, sizeof(pattern) - 1, 5.0f, 0.5f, 0.1f, "code_slice");
}



int run_elm_overfit_tests(void) {
    int ok = 1;
    ok &= test_overfit_batch_pattern_a();
    ok &= test_overfit_batch_pattern_b();
    ok &= test_overfit_batch_code_slice();
    return ok;
}