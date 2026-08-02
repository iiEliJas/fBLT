#include <stdio.h>
#include "blt/core/allocator.h"
#include "blt/models/entropy.h"

int run_entropy_model_tests(void) {
    blt_arena* arena = blt_arena_create(4096, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "[FAIL] arena creation for entropy tests\n");
        return 0;
    }

    size_t probs_shape[2] = {2, 3};
    size_t entropy_shape[1] = {2};
    blt_tensor probs = blt_tensor_create(arena, probs_shape, 2, BLT_DTYPE_FP32);
    blt_tensor entropy = blt_tensor_create(arena, entropy_shape, 1, BLT_DTYPE_FP32);

    float* probs_data = (float*)probs.data;
    float* entropy_data = (float*)entropy.data;
    probs_data[0] = 0.5f; probs_data[1] = 0.25f; probs_data[2] = 0.25f;
    probs_data[3] = 1.0f; probs_data[4] = 0.0f; probs_data[5] = 0.0f;

    blt_entropy_config config = {0};
    config.vocab_size = 3;
    config.use_log2 = true;

    blt_compute_entropy(&probs, &entropy, &config);

    if (entropy_data[0] <= 0.0f || entropy_data[1] != 0.0f) {
        fprintf(stderr, "[FAIL] entropy computation\n");
        blt_arena_destroy(arena);
        return 0;
    }

    blt_arena_destroy(arena);
    return 1;
}
