#include "test_helpers.h"
#include "test_suite.h"
#include "blt/ops/mask_builder.h"
#include "blt/models/patcher.h"
#include "blt/models/local_decoder.h"
#include "blt/ops/optim.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

//------------------------------------------------------------------------
// Local Decoder Tests
//
// The local decoders cross-attention mask is a [seq_len, num_patches] matrix
// where each row corresponds to a byte and each column corresponds to a patch
// Each byte should only attend to its own patch, so the mask should have exactly one non -INFINITY value per row,
// and that value should be in the column corresponding to the patch that contains that byte.
// This test verifies that the mask is constructed correctly for a simple set of patches and bytes.

static void build_decoder_test_patches(blt_patch_info *patches, size_t *num_patches, size_t *seq_len) {
    size_t starts[] = {0, 3, 7, 8, 15};
    size_t n = sizeof(starts) / sizeof(starts[0]);
    size_t total = 20;
    for (size_t i = 0; i < n; i++) {
        patches[i].start_idx = starts[i];
        patches[i].length = (i + 1 < n) ? (starts[i + 1] - starts[i]) : (total - starts[i]);
        patches[i].peak_entropy = 0.0f;
    }
    *num_patches = n;
    *seq_len = total;
}

int run_local_decoder_cross_mask(void) {
    blt_arena *arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    if (!arena) return 0;

    blt_patch_info patches[8];
    size_t num_patches = 0, seq_len = 0;
    build_decoder_test_patches(patches, &num_patches, &seq_len);

    size_t *patch_identity_ids = (size_t *)blt_arena_alloc(arena, num_patches * sizeof(size_t), 64);
    size_t *byte_patch_ids = (size_t *)blt_arena_alloc(arena, seq_len * sizeof(size_t), 64);
    TEST_ASSERT(patch_identity_ids != NULL && byte_patch_ids != NULL);
    blt_patch_build_group_ids(patches, num_patches, seq_len, patch_identity_ids, byte_patch_ids);

    blt_mask_config cross_mask = {0};
    cross_mask.seq_len_q = seq_len;
    cross_mask.seq_len_kv = num_patches;
    cross_mask.query_group_ids = byte_patch_ids;  // bytes own patch id
    cross_mask.kv_group_ids = patch_identity_ids; // patch j group id = j
    cross_mask.bidirectional_within_group = true;
    cross_mask.is_causal = false;

    blt_tensor mask = {0};
    blt_build_attention_mask(&cross_mask, &mask, arena);
    TEST_ASSERT(mask.ndim == 2 && mask.shape[0] == seq_len && mask.shape[1] == num_patches);

    const float *data = (const float *)mask.data;

    // Every byte row must have its own patch column as the only non-masked value
    for (size_t i = 0; i < seq_len; i++) {
        int allowed_count = 0;
        size_t allowed_col = (size_t)-1;
        for (size_t j = 0; j < num_patches; j++) {
            float v = data[i * num_patches + j];
            if (v > -INFINITY) {
                allowed_count++;
                allowed_col = j;
            }
        }
        TEST_ASSERT(allowed_count == 1);
        TEST_ASSERT(allowed_col == byte_patch_ids[i]);
    }

    // Every patch must be the non-masked column for at least one byte row
    bool patch_hit[8] = {0};
    for (size_t i = 0; i < seq_len; i++) {
        patch_hit[byte_patch_ids[i]] = true;
    }
    for (size_t j = 0; j < num_patches; j++) {
        TEST_ASSERT(patch_hit[j]);
    }

    blt_arena_destroy(arena);
    return 1;
}

//------------------------------------------------------------------------
// Local Decoder overfit test
//
// This test creates a small local decoder model and trains it on a single batch of random data for multiple steps,
// verifying that the loss decreases significantly over time.
// This ensures that the forward and backward passes are functioning correctly
// and that the model can learn from the data.

static void fill_random(blt_tensor *t, float scale) {
    float *d = (float *)t->data;
    for (size_t i = 0; i < t->numel; i++) {
        d[i] = scale * (((float)(rand() % 2000) / 1000.0f) - 1.0f);
    }
}

static void fill_random_bytes(blt_tensor *t) {
    uint8_t *d = (uint8_t *)t->data;
    for (size_t i = 0; i < t->numel; i++) {
        d[i] = (uint8_t)(rand() % 256);
    }
}

static void sgd_update_local_decoder(blt_local_decoder *model, const blt_local_decoder_grad *grad, float lr) {
    for (size_t l = 0; l < model->config.num_layers; l++) {
        blt_local_decoder_layer_storage *s = &model->layers[l];
        const blt_local_decoder_layer_grad *g = &grad->layer_grads[l];

        blt_sgd_step(&s->norm1_weight, &g->norm1_weight, lr);
        blt_sgd_step(&s->attn_qkv_w, &g->attn_qkv_w, lr);
        blt_sgd_step(&s->attn_proj_w, &g->attn_proj_w, lr);
        blt_sgd_step(&s->norm2_weight, &g->norm2_weight, lr);
        blt_sgd_step(&s->ffn_up_w, &g->ffn_up_w, lr);
        blt_sgd_step(&s->ffn_gate_w, &g->ffn_gate_w, lr);
        blt_sgd_step(&s->ffn_down_w, &g->ffn_down_w, lr);

        blt_sgd_step(&s->cross_norm_weight, &g->cross_norm_weight, lr);
        blt_sgd_step(&s->cross_weight_q, &g->cross_weight_q, lr);
        blt_sgd_step(&s->cross_weight_k, &g->cross_weight_k, lr);
        blt_sgd_step(&s->cross_weight_v, &g->cross_weight_v, lr);
        blt_sgd_step(&s->cross_weight_proj, &g->cross_weight_proj, lr);
    }
    blt_sgd_step(&model->lm_head_weight, &grad->lm_head_grad, lr);
}

int run_local_decoder_overfit(void) {
    // arena for model and gradients
    blt_arena *arena = blt_arena_create(16 * 1024 * 1024, BLT_BACKEND_CPU);
    // arena for intermediate tensors
    blt_arena *compute_arena = blt_arena_create(16 * 1024 * 1024, BLT_BACKEND_CPU);

    if (!arena || !compute_arena) return 0;

    srand(42);

    blt_patch_info patches[3];
    patches[0].start_idx = 0;
    patches[0].length = 3;
    patches[0].peak_entropy = 0.0f;
    patches[1].start_idx = 3;
    patches[1].length = 3;
    patches[1].peak_entropy = 0.0f;
    patches[2].start_idx = 6;
    patches[2].length = 2;
    patches[2].peak_entropy = 0.0f;
    size_t num_patches = 3;
    size_t seq_len = 8;

    blt_local_decoder_config cfg = {0};
    cfg.embed_dim = 16;
    cfg.num_layers = 2;
    cfg.hidden_dim = 32;
    cfg.num_heads = 4;
    cfg.cross_attn_heads = 4;
    cfg.local_window = 0; // full causal
    cfg.cross_attn_all_layers = true;
    cfg.rope_theta = 500000.0f;
    cfg.max_seq_len = 32;
    cfg.vocab_size = 256;

    blt_local_decoder *model = blt_local_decoder_create(arena, &cfg);
    TEST_ASSERT(model != NULL);

    for (size_t l = 0; l < cfg.num_layers; l++) {
        blt_local_decoder_layer_storage *s = &model->layers[l];
        fill_random(&s->norm1_weight, 1.0f);
        fill_random(&s->attn_qkv_w, 0.1f);
        fill_random(&s->attn_proj_w, 0.1f);
        fill_random(&s->norm2_weight, 1.0f);
        fill_random(&s->ffn_up_w, 0.1f);
        fill_random(&s->ffn_gate_w, 0.1f);
        fill_random(&s->ffn_down_w, 0.1f);
        fill_random(&s->cross_norm_weight, 1.0f);
        fill_random(&s->cross_weight_q, 0.1f);
        fill_random(&s->cross_weight_k, 0.1f);
        fill_random(&s->cross_weight_v, 0.1f);
        fill_random(&s->cross_weight_proj, 0.1f);
    }
    fill_random(&model->lm_head_weight, 0.1f);

    blt_local_decoder_grad *grad = blt_local_decoder_grad_create(arena, model);
    TEST_ASSERT(grad != NULL);

    size_t byte_shape[2] = {seq_len, cfg.embed_dim};
    size_t patch_shape[2] = {num_patches, cfg.embed_dim};
    blt_tensor byte_hidden_in = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_tensor patch_in = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    fill_random(&byte_hidden_in, 1.0f);
    fill_random(&patch_in, 1.0f);

    size_t bytes_shape[1] = {seq_len};
    blt_tensor bytes_in = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
    fill_random_bytes(&bytes_in);

    size_t logits_shape[2] = {seq_len, cfg.vocab_size};
    size_t loss_shape[1] = {1};
    blt_tensor logits_out = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_tensor loss_out = blt_tensor_create(arena, loss_shape, 1, BLT_DTYPE_FP32);
    blt_tensor grad_byte_hidden_in = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_tensor grad_patch_in = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);

    float first_loss = 0.0f, last_loss = 0.0f;
    const int num_steps = 200;
    const float lr = 0.05f;

    for (int step = 0; step < num_steps; step++) {
        blt_arena_reset(compute_arena);

        blt_local_decoder_forward(model, &byte_hidden_in, &patch_in, patches, num_patches, &bytes_in, NULL, 0,
                                  &logits_out, &loss_out, compute_arena);
        float loss_val = ((float *)loss_out.data)[0];
        if (step == 0) first_loss = loss_val;
        last_loss = loss_val;

        blt_local_decoder_backward(model, &byte_hidden_in, &patch_in, patches, num_patches, &bytes_in, NULL, 0,
                                   &grad_byte_hidden_in, &grad_patch_in, grad, compute_arena);

        sgd_update_local_decoder(model, grad, lr);
    }

    TEST_ASSERT(last_loss < first_loss * 0.5f);

    blt_arena_destroy(arena);
    blt_arena_destroy(compute_arena);
    return 1;
}

//------------------------------------------------------------------------
// Local Decoder k-split test
//
// Same shape as the overfit test, but with patch_dim = k * embed_dim (k=3),
// so patch_in is wider than embed_dim and cross-attention has to go through
// the split-kv path (BLT_CROSS_ATTN_SPLIT_KV) instead of the k=1 passthrough.

int run_local_decoder_k_split(void) {
    blt_arena *arena = blt_arena_create(16 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *compute_arena = blt_arena_create(16 * 1024 * 1024, BLT_BACKEND_CPU);
    if (!arena || !compute_arena) return 0;

    srand(7);

    blt_patch_info patches[3];
    patches[0].start_idx = 0;
    patches[0].length = 3;
    patches[0].peak_entropy = 0.0f;
    patches[1].start_idx = 3;
    patches[1].length = 3;
    patches[1].peak_entropy = 0.0f;
    patches[2].start_idx = 6;
    patches[2].length = 2;
    patches[2].peak_entropy = 0.0f;
    size_t num_patches = 3;
    size_t seq_len = 8;

    size_t k = 3;
    size_t E = 16;

    blt_local_decoder_config cfg = {0};
    cfg.embed_dim = E;
    cfg.patch_dim = E * k; // = 48, forces the split path
    cfg.num_layers = 2;
    cfg.hidden_dim = 32;
    cfg.num_heads = 4;
    cfg.cross_attn_heads = 4;
    cfg.local_window = 0; // full causal
    cfg.cross_attn_all_layers = true;
    cfg.rope_theta = 500000.0f;
    cfg.max_seq_len = 32;
    cfg.vocab_size = 256;

    blt_local_decoder *model = blt_local_decoder_create(arena, &cfg);
    TEST_ASSERT(model != NULL);

    for (size_t l = 0; l < cfg.num_layers; l++) {
        blt_local_decoder_layer_storage *s = &model->layers[l];
        fill_random(&s->norm1_weight, 1.0f);
        fill_random(&s->attn_qkv_w, 0.1f);
        fill_random(&s->attn_proj_w, 0.1f);
        fill_random(&s->norm2_weight, 1.0f);
        fill_random(&s->ffn_up_w, 0.1f);
        fill_random(&s->ffn_gate_w, 0.1f);
        fill_random(&s->ffn_down_w, 0.1f);
        fill_random(&s->cross_norm_weight, 1.0f);
        fill_random(&s->cross_weight_q, 0.1f);
        fill_random(&s->cross_weight_k, 0.1f);
        fill_random(&s->cross_weight_v, 0.1f);
        fill_random(&s->cross_weight_proj, 0.1f);
    }
    fill_random(&model->lm_head_weight, 0.1f);

    blt_local_decoder_grad *grad = blt_local_decoder_grad_create(arena, model);
    TEST_ASSERT(grad != NULL);

    size_t byte_shape[2] = {seq_len, E};
    size_t patch_shape[2] = {num_patches, cfg.patch_dim}; // patch_dim-wide, not E-wide
    blt_tensor byte_hidden_in = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_tensor patch_in = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    fill_random(&byte_hidden_in, 1.0f);
    fill_random(&patch_in, 1.0f);

    size_t bytes_shape[1] = {seq_len};
    blt_tensor bytes_in = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
    fill_random_bytes(&bytes_in);

    size_t logits_shape[2] = {seq_len, cfg.vocab_size};
    size_t loss_shape[1] = {1};
    blt_tensor logits_out = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_tensor loss_out = blt_tensor_create(arena, loss_shape, 1, BLT_DTYPE_FP32);
    blt_tensor grad_byte_hidden_in = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_tensor grad_patch_in = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);

    float first_loss = 0.0f, last_loss = 0.0f;
    const int num_steps = 200;
    const float lr = 0.05f;

    for (int step = 0; step < num_steps; step++) {
        blt_arena_reset(compute_arena);

        blt_local_decoder_forward(model, &byte_hidden_in, &patch_in, patches, num_patches, &bytes_in, NULL, 0,
                                  &logits_out, &loss_out, compute_arena);
        float loss_val = ((float *)loss_out.data)[0];
        TEST_ASSERT(!isnan(loss_val) && !isinf(loss_val));
        if (step == 0) first_loss = loss_val;
        last_loss = loss_val;

        blt_local_decoder_backward(model, &byte_hidden_in, &patch_in, patches, num_patches, &bytes_in, NULL, 0,
                                   &grad_byte_hidden_in, &grad_patch_in, grad, compute_arena);

        // grad_patch_in must come back at the full patch_dim width, not E
        TEST_ASSERT(grad_patch_in.shape[0] == num_patches && grad_patch_in.shape[1] == cfg.patch_dim);

        sgd_update_local_decoder(model, grad, lr);
    }

    TEST_ASSERT(last_loss < first_loss * 0.5f);

    blt_arena_destroy(arena);
    blt_arena_destroy(compute_arena);
    return 1;
}