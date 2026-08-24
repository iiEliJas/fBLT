#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "blt/core/allocator.h"
#include "blt/core/tensor.h"
#include "blt/models/entropy.h"
#include "blt/models/patcher.h"

// Loader for FP32 tensors
static int load_binary_tensor(const char* path, blt_arena* arena, blt_tensor* out_tensor) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "failed to open %s\n", path);
        return 0;
    }

    uint32_t ndim = 0;
    if (fread(&ndim, sizeof(ndim), 1, fp) != 1) {
        fclose(fp);
        return 0;
    }

    size_t shape[BLT_MAX_NDIM] = {0};
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t dim = 0;
        if (fread(&dim, sizeof(dim), 1, fp) != 1) {
            fclose(fp);
            return 0;
        }
        shape[i] = dim;
    }

    size_t numel = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        numel *= shape[i];
    }

    *out_tensor = blt_tensor_create(arena, shape, ndim, BLT_DTYPE_FP32);
    if (!out_tensor->data) {
        fclose(fp);
        return 0;
    }

    float* data = (float*)out_tensor->data;
    for (size_t i = 0; i < numel; ++i) {
        float value = 0.0f;
        if (fread(&value, sizeof(value), 1, fp) != 1) {
            fclose(fp);
            return 0;
        }
        data[i] = value;
    }

    fclose(fp);
    return 1;
}

// Loader for UINT8 Tensors (Used for bytes array)
static int load_binary_tensor_uint8(const char* path, blt_arena* arena, blt_tensor* out_tensor) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "failed to open %s\n", path);
        return 0;
    }

    uint32_t ndim = 0;
    if (fread(&ndim, sizeof(ndim), 1, fp) != 1) {
        fclose(fp);
        return 0;
    }

    size_t shape[BLT_MAX_NDIM] = {0};
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t dim = 0;
        if (fread(&dim, sizeof(dim), 1, fp) != 1) {
            fclose(fp);
            return 0;
        }
        shape[i] = dim;
    }

    size_t numel = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        numel *= shape[i];
    }

    *out_tensor = blt_tensor_create(arena, shape, ndim, BLT_DTYPE_UINT8);
    if (!out_tensor->data) {
        fclose(fp);
        return 0;
    }

    uint8_t* data = (uint8_t*)out_tensor->data;
    if (fread(data, 1, numel, fp) != numel) {
        fclose(fp);
        return 0;
    }

    fclose(fp);
    return 1;
}

// Loader for reference boundary logic
static int load_patch_boundaries(const char* path, size_t* starts, size_t* lengths, size_t max_patches, size_t* out_count) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "failed to open %s\n", path);
        return 0;
    }

    char line[128];
    size_t count = 0;
    while (fgets(line, sizeof(line), fp) && count < max_patches) {
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }

        size_t start = 0;
        size_t length = 0;
        float peak = 0.0f;
        if (sscanf(line, "%zu %zu %f", &start, &length, &peak) == 3) {
            starts[count] = start;
            lengths[count] = length;
            count++;
        }
    }

    fclose(fp);
    *out_count = count;
    return 1;
}



int run_patcher_parity_tests(void) {
    blt_arena* arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "[FAIL] arena creation for patcher tests\n");
        return 0;
    }

    blt_tensor probs = {0};
    blt_tensor expected_entropy = {0};
    blt_tensor bytes_tensor = {0};

    // Load probabilities, entropy, AND the generated uint8 byte stream
    if (!load_binary_tensor("data/patcher_probs.bin", arena, &probs) ||
        !load_binary_tensor("data/patcher_entropy.bin", arena, &expected_entropy) ||
        !load_binary_tensor_uint8("data/patcher_bytes.bin", arena, &bytes_tensor)) {
        blt_arena_destroy(arena);
        return 0;
    }

    if (probs.ndim != 2 || expected_entropy.ndim != 1 || expected_entropy.numel != probs.shape[0]) {
        fprintf(stderr, "[FAIL] invalid reference tensor shapes\n");
        blt_arena_destroy(arena);
        return 0;
    }

    blt_tensor computed_entropy = blt_tensor_create(arena, expected_entropy.shape, expected_entropy.ndim, BLT_DTYPE_FP32);
    blt_entropy_config config = {0};
    config.vocab_size = probs.shape[1];
    config.use_log2 = true;

    blt_patcher_config p_config = {0};
    p_config.threshold_global = 0.6f;
    p_config.threshold_monotonic = 0.4f;
    p_config.max_patch_length = 5;
    p_config.rule = BLT_PATCH_RULE_BOTH;
    p_config.reset_on_newline = true;

    blt_compute_entropy(&probs, &computed_entropy, &config);

    const float* got_entropy = (const float*)computed_entropy.data;
    const float* want_entropy = (const float*)expected_entropy.data;
    for (size_t i = 0; i < computed_entropy.numel; ++i) {
        float diff = fabsf(got_entropy[i] - want_entropy[i]);
        if (diff > 1e-4f) {
            fprintf(stderr, "[FAIL] entropy mismatch at %zu: got %.6f expected %.6f\n", i, got_entropy[i], want_entropy[i]);
            blt_arena_destroy(arena);
            return 0;
        }
    }

    size_t max_patches = 1024;
    blt_patch_info* patches_out = (blt_patch_info*)blt_arena_alloc(arena, sizeof(blt_patch_info) * max_patches, sizeof(void*));
    if (!patches_out) {
        fprintf(stderr, "[FAIL] patch output allocation\n");
        blt_arena_destroy(arena);
        return 0;
    }

    // Pass the raw byte array data into segment patches[cite: 7]
    const uint8_t* raw_bytes = (const uint8_t*)bytes_tensor.data;
    size_t num_patches = blt_segment_patches(&computed_entropy, raw_bytes, patches_out, max_patches, &p_config);
    
    if (num_patches == 0) {
        fprintf(stderr, "[FAIL] patch segmentation produced zero patches\n");
        blt_arena_destroy(arena);
        return 0;
    }

    size_t expected_starts[1024];
    size_t expected_lengths[1024];
    size_t expected_count = 0;
    if (!load_patch_boundaries("data/patcher_boundaries.txt", expected_starts, expected_lengths, 1024, &expected_count)) {
        blt_arena_destroy(arena);
        return 0;
    }

    if (expected_count != num_patches) {
        fprintf(stderr, "[FAIL] patch count mismatch: got %zu expected %zu\n", num_patches, expected_count);
        blt_arena_destroy(arena);
        return 0;
    }

    for (size_t i = 0; i < num_patches; ++i) {
        if (patches_out[i].start_idx != expected_starts[i] || patches_out[i].length != expected_lengths[i]) {
            fprintf(stderr, "[FAIL] patch[%zu] mismatch: got start=%zu length=%zu expected start=%zu length=%zu\n",
                    i, patches_out[i].start_idx, patches_out[i].length, expected_starts[i], expected_lengths[i]);
            blt_arena_destroy(arena);
            return 0;
        }
    }

    blt_arena_destroy(arena);
    return 1;
}