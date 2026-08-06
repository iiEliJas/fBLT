#include <stdio.h>
#include "blt/core/allocator.h"
#include "blt/models/entropy.h"
#include "blt/models/patcher.h"

#include "test_helpers.h"
#include "test_suite.h"

int run_patcher_model_tests(void) {
    blt_arena* arena = blt_arena_create(4096, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "[FAIL] arena creation for patcher tests\n");
        return 0;
    }

    size_t entropy_shape[1] = {10};
    blt_tensor entropy = blt_tensor_create(arena, entropy_shape, 1, BLT_DTYPE_FP32);

    float* entropy_data = (float*)entropy.data;
    // Simulate some entropy values
    entropy_data[0] = 0.1f; entropy_data[1] = 0.2f; entropy_data[2] = 0.5f; 
    entropy_data[3] = 0.9f; entropy_data[4] = 0.8f; entropy_data[5] = 0.3f; 
    entropy_data[6] = 0.2f; entropy_data[7] = 0.6f; entropy_data[8] = 0.7f; 
    entropy_data[9] = 0.1f;

    blt_entropy_config config = {0};
    config.threshold = 0.5f;

    blt_patch_info patches_out[10];
    size_t num_patches = blt_segment_patches(&entropy, patches_out, 10, &config);

    TEST_ASSERT(num_patches != 0);

    // Optionally print the patches for verification
    for (size_t i = 0; i < num_patches; ++i) {
        printf("Patch %zu: start_idx=%zu, length=%zu, peak_entropy=%.2f\n", 
               i, patches_out[i].start_idx, patches_out[i].length, patches_out[i].peak_entropy);
    }

    blt_arena_destroy(arena);
    return 1;
}