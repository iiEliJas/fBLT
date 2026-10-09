#include "core/allocator.h"
#include "models/model.h"
#include "models/entropy_lm.h"
#include "models/patcher.h"
#include "core/backend.h"
#include "models/entropy.h"
#include "ops/softmax.h"
#include "core/generate_greedy.h"

#include <stdint.h>

#define BLT_GENERATE_MAX_PATCHES 4096

// opts may be NULL, which means argmax at every step. History for the penalty
// and ngram ban is the whole emitted sequence including the prompt.
static void generate_loop(const blt_model *model, const blt_entropy_lm *entropy_model,
                          const blt_patcher_config *patcher_config, const uint8_t *prompt_bytes, size_t prompt_len,
                          size_t max_new_bytes, uint8_t *output_bytes, const blt_decode_options *opts,
                          blt_arena *arena) {
    BLT_REQUIRE(model != NULL && entropy_model != NULL && patcher_config != NULL && prompt_bytes != NULL &&
                    output_bytes != NULL && arena != NULL,
                "blt_generate_greedy: arguments cannot be NULL");
    BLT_REQUIRE(prompt_len >= 2, "blt_generate_greedy: prompt_len must be >= 2");
    if (opts) {
        BLT_REQUIRE(opts->temperature >= 0.0f, "blt_generate_greedy: temperature must be >= 0");
        BLT_REQUIRE(opts->repeat_penalty > 0.0f, "blt_generate_greedy: repeat_penalty must be > 0");
    }
    uint64_t rng = opts ? (opts->seed ? opts->seed : 1) : 1;

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

        // Trailing patch closure: a boundary at cur_len is decidable from the
        // prefix (entropy_data[cur_len-1], bytes[cur_len-1], never the byte we
        // are about to predict). If it fires, byte cur_len-1 is final and reads
        // its own latent, matching the paper's prefix-decidable rule.
        int trailing_closed = 0;
        if (num_patches >= 1) {
            const size_t ls = patches[num_patches - 1].start_idx;
            const size_t ll = patches[num_patches - 1].length;
            trailing_closed = blt_next_starts_patch(output_bytes, cur_len, entropy_host, ls, ll, patcher_config);
        }
        free(entropy_host);

        // 3. Full encoder-global-decoder forward pass (single document).
        // Only the last row's latent depends on trailing_closed, so the encode
        // stage is shared and the decode is called with the flag directly.
        size_t vocab_shape[2] = {cur_len, vocab_size};
        blt_tensor model_logits = blt_tensor_create(arena, vocab_shape, 2, BLT_DTYPE_FP32);

        blt_model_enc_out enc;
        blt_model_encode(model, &bytes_in, patches, num_patches, NULL, 0, &enc, arena);
        blt_local_decoder_forward_ext(model->decoder, &enc.byte_hidden_out, &enc.global_out, patches, num_patches, NULL,
                                      NULL, NULL, 0, NULL, trailing_closed, &model_logits, NULL, arena);
        blt_backend_pass_sync(bytes_in.backend);

        // 4. Pick the last position's next byte.
        float *last_row = (float *)malloc(vocab_size * sizeof(float));
        BLT_REQUIRE(last_row != NULL, "blt_generate_greedy: staging alloc failed");
        blt_tensor last_row_view;
        view_1d(&last_row_view, (float *)model_logits.data + (cur_len - 1) * vocab_size, vocab_size, BLT_DTYPE_FP32,
                model_logits.backend);
        blt_tensor_download(&last_row_view, last_row, vocab_size * sizeof(float));
        uint8_t next_byte = blt_decode_select(opts, last_row, vocab_size, output_bytes, cur_len, &rng);
        free(last_row);

        output_bytes[cur_len] = next_byte;
        cur_len++;

        arena->offset = step_marker;
    }
}
void blt_generate_greedy(const blt_model *model, const blt_entropy_lm *entropy_model,
                         const blt_patcher_config *patcher_config, const uint8_t *prompt_bytes, size_t prompt_len,
                         size_t max_new_bytes, uint8_t *output_bytes, blt_arena *arena) {
    generate_loop(model, entropy_model, patcher_config, prompt_bytes, prompt_len, max_new_bytes, output_bytes, NULL,
                  arena);
}

void blt_generate_sample(const blt_model *model, const blt_entropy_lm *entropy_model,
                         const blt_patcher_config *patcher_config, const uint8_t *prompt_bytes, size_t prompt_len,
                         size_t max_new_bytes, uint8_t *output_bytes, const blt_decode_options *opts,
                         blt_arena *arena) {
    BLT_REQUIRE(opts != NULL, "blt_generate_sample: opts cannot be NULL");
    BLT_REQUIRE(blt_decode_is_sampling(opts),
                "blt_generate_sample: no sampling knob enabled; use blt_generate_greedy instead");
    generate_loop(model, entropy_model, patcher_config, prompt_bytes, prompt_len, max_new_bytes, output_bytes, opts,
                  arena);
}
