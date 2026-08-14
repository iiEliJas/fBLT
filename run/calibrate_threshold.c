#include "blt/models/patcher.h"
#include "blt/core/backend.h"
#include "blt/core/allocator.h"
#include "blt/core/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    size_t total_bytes;
    size_t total_patches;
    size_t min_patch_len;
    size_t max_patch_len;
    float avg_patch_len;
} calibration_stats;

static calibration_stats compute_stats(const blt_patch_info* patches, size_t patch_count, size_t total_bytes){
    calibration_stats stats = {
        .total_bytes = total_bytes,
        .total_patches = patch_count,
        .min_patch_len = patch_count > 0 ? patches[0].length : 0,
        .max_patch_len = 0,
        .avg_patch_len = patch_count > 0 ? (float)total_bytes / (float)patch_count : 0.0f
    };

    for(size_t i = 0; i < patch_count; ++i){
        if(patches[i].length < stats.min_patch_len){
            stats.min_patch_len = patches[i].length;
        }
        if(patches[i].length > stats.max_patch_len){
            stats.max_patch_len = patches[i].length;
        }
    }

    return stats;
}



int main(int argc, char** argv){
    if(argc < 2){
        fprintf(stderr, "Usage: %s <entropy_file> [target_patch_size]\n", argv[0]);
        fprintf(stderr, "  entropy_file: path to binary FP32 entropy array\n");
        fprintf(stderr, "  target_patch_size: desired average patch length (default 6.0)\n");
        return 1;
    }

    float target_patch_size = 6.0f;
    if(argc >= 3){
        target_patch_size = atof(argv[2]);
    }

    FILE* f = fopen(argv[1], "rb");
    if(!f){
        fprintf(stderr, "Error: cannot open %s\n", argv[1]);
        return 1;
    }

    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if(file_size % sizeof(float) != 0){
        fprintf(stderr, "Error: file size not a multiple of sizeof(float)\n");
        fclose(f);
        return 1;
    }

    size_t seq_len = file_size / sizeof(float);

    blt_arena* arena = blt_arena_create(file_size, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "Error: arena creation for patcher tests\n");
        return 0;
    }

    size_t entropy_shape[1] = {seq_len};
    blt_tensor entropy_tensor = blt_tensor_create(arena, entropy_shape, 1, BLT_DTYPE_FP32);
    float* entropy_data = (float*)entropy_tensor.data;

    if(fread(entropy_data, sizeof(float), seq_len, f) != seq_len){
        fprintf(stderr, "Error: read failed\n");
        free(entropy_data);
        fclose(f);
        return 1;
    }
    fclose(f);


    size_t max_patches = seq_len;
    blt_patch_info* patches = (blt_patch_info*)malloc(max_patches * sizeof(blt_patch_info));
    if(!patches){
        fprintf(stderr, "Error: allocation failed\n");
        free(entropy_data);
        return 1;
    }

    float theta_low = 0.0f;
    float theta_high = 0.0f;
    for(size_t i = 0; i < seq_len; ++i){
        if(entropy_data[i] > theta_high){
            theta_high = entropy_data[i];
        }
    }

    float epsilon = 0.01f;
    float theta_best = theta_low;
    float avg_best = 0.0f;
    int max_iter = 40;

    printf("\nSearching for threshold...\n");
    printf("Target patch size: %.2f, tolerance: %.4f\n", target_patch_size, epsilon);

    for(int iter = 0; iter < max_iter; ++iter){
        float theta_mid = (theta_low + theta_high) / 2.0f;

        blt_patcher_config config = {
            .threshold_global = theta_mid,
            .threshold_monotonic = 0.0f,
            .max_patch_length = 256,
            .rule = BLT_PATCH_RULE_GLOBAL,
            .reset_on_newline = false
        };

        size_t patch_count = blt_segment_patches(&entropy_tensor, NULL, patches, max_patches, &config);
        float avg_len = (float)seq_len / (float)patch_count;

        printf("Iter %2d: theta=%.6f, avg_patch_len=%.3f\n", iter, theta_mid, avg_len);

        if(fabsf(avg_len - target_patch_size) < epsilon){
            theta_best = theta_mid;
            avg_best = avg_len;
            break;
        }

        theta_best = theta_mid;
        avg_best = avg_len;

        if(avg_len < target_patch_size){
            theta_low = theta_mid;
        } else {
            theta_high = theta_mid;
        }
    }

    calibration_stats stats = compute_stats(patches, seq_len, seq_len);
    if(stats.total_patches > 0){
        blt_patcher_config config = {
            .threshold_global = theta_best,
            .threshold_monotonic = 0.0f,
            .max_patch_length = 256,
            .rule = BLT_PATCH_RULE_GLOBAL,
            .reset_on_newline = false
        };
        size_t final_patch_count = blt_segment_patches(&entropy_tensor, NULL, patches, max_patches, &config);
        stats = compute_stats(patches, final_patch_count, seq_len);
    }

    printf("\n--- Calibration Results ---n");
    printf("Calibrated threshold:  %.6f\n", theta_best);
    printf("Total bytes:           %zu\n", stats.total_bytes);
    printf("Total patches:         %zu\n", stats.total_patches);
    printf("Average patch length:  %.3f\n", stats.avg_patch_len);
    printf("Min patch length:      %zu\n", stats.min_patch_len);
    printf("Max patch length:      %zu\n", stats.max_patch_len);

    free(patches);
    blt_arena_destroy(arena);

    return 0;
}