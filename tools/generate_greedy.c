#include "blt/core/allocator.h"
#include "blt/models/model.h"
#include "blt/models/entropy_lm.h"
#include "blt/models/patcher.h"
#include "blt/core/backend.h"
#include "blt/models/entropy.h"
#include "blt/ops/softmax.h"
#include "generate_greedy.h"

#include <stdint.h>

#define BLT_GENERATE_MAX_PATCHES 4096

static uint8_t argmax_byte(const float *row, size_t vocab_size) {
    size_t best = 0;
    float best_val = row[0];
    for (size_t v = 1; v < vocab_size; v++) {
        if (row[v] > best_val) {
            best_val = row[v];
            best = v;
        }
    }
    return (uint8_t)best;
}

void blt_generate_greedy(const blt_model *model, const blt_entropy_lm *entropy_model,
                         const blt_patcher_config *patcher_config, const uint8_t *prompt_bytes, size_t prompt_len,
                         size_t max_new_bytes, uint8_t *output_bytes, blt_arena *arena) {
    BLT_REQUIRE(model != NULL && entropy_model != NULL && patcher_config != NULL && prompt_bytes != NULL &&
                    output_bytes != NULL && arena != NULL,
                "blt_generate_greedy: arguments cannot be NULL");
    BLT_REQUIRE(prompt_len >= 2, "blt_generate_greedy: prompt_len must be >= 2");

    memcpy(output_bytes, prompt_bytes, prompt_len);
    size_t cur_len = prompt_len;
    size_t target_len = prompt_len + max_new_bytes;
    size_t vocab_size = model->config.decoder_config.vocab_size;

    while (cur_len < target_len) {
        size_t step_marker = arena->offset;

        // 1. Entropy model over the current byte sequence
        size_t bytes_shape[1] = {cur_len};
        blt_tensor bytes_in = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
        blt_tensor_upload(&bytes_in, output_bytes, cur_len);

        size_t entropy_logits_shape[2] = {cur_len, 256};
        blt_tensor entropy_logits = blt_tensor_create(arena, entropy_logits_shape, 2, BLT_DTYPE_FP32);
        size_t scalar_shape[1] = {1};
        blt_tensor entropy_loss = blt_tensor_create(arena, scalar_shape, 1, BLT_DTYPE_FP32);

        blt_entropy_lm_forward(entropy_model, &bytes_in, &entropy_logits, &entropy_loss, arena);

        blt_tensor probs = blt_tensor_create(arena, entropy_logits_shape, 2, BLT_DTYPE_FP32);
        blt_softmax(&entropy_logits, &probs);

        size_t entropy_shape[1] = {cur_len};
        blt_tensor entropy_vals = blt_tensor_create(arena, entropy_shape, 1, BLT_DTYPE_FP32);
        blt_entropy_config entropy_cfg = {.vocab_size = 256, .use_log2 = false};
        blt_compute_entropy(&probs, &entropy_vals, &entropy_cfg);

        // 2. Segment the current sequence into patches.
        //
        // The patcher is host-side control logic, so it consumes a host
        // copy of the entropy signal regardless of the model backend.
        // Host staging (plain malloc: the scratch arena may be device
        // memory, which is not host-writable).
        float *entropy_host = (float *)malloc(cur_len * sizeof(float));
        BLT_REQUIRE(entropy_host != NULL, "blt_generate_greedy: staging alloc failed");
        blt_tensor_download(&entropy_vals, entropy_host, cur_len * sizeof(float));
        blt_tensor entropy_host_view;
        view_1d(&entropy_host_view, entropy_host, cur_len, BLT_DTYPE_FP32, BLT_BACKEND_CPU);

        blt_patch_info patches[BLT_GENERATE_MAX_PATCHES];
        size_t num_patches =
            blt_segment_patches(&entropy_host_view, output_bytes, patches, BLT_GENERATE_MAX_PATCHES, patcher_config);
        free(entropy_host);

        // 3. Full encoder-global-decoder forward pass (single document)
        size_t vocab_shape[2] = {cur_len, vocab_size};
        blt_tensor model_logits = blt_tensor_create(arena, vocab_shape, 2, BLT_DTYPE_FP32);

        blt_tensor discard_loss = blt_tensor_create(arena, scalar_shape, 1, BLT_DTYPE_FP32);
        blt_model_forward(model, &bytes_in, patches, num_patches, NULL, 0, &model_logits, &discard_loss, arena);

        // 4. Greedy argmax of the last position's logits
        float *last_row = (float *)malloc(vocab_size * sizeof(float));
        BLT_REQUIRE(last_row != NULL, "blt_generate_greedy: staging alloc failed");
        blt_tensor last_row_view;
        view_1d(&last_row_view, (float *)model_logits.data + (cur_len - 1) * vocab_size, vocab_size, BLT_DTYPE_FP32,
                model_logits.backend);
        blt_tensor_download(&last_row_view, last_row, vocab_size * sizeof(float));
        uint8_t next_byte = argmax_byte(last_row, vocab_size);
        free(last_row);

        output_bytes[cur_len] = next_byte;
        cur_len++;

        arena->offset = step_marker;
    }
}