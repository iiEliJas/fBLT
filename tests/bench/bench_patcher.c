#include "models/patcher.h"
#include "core/allocator.h"
#include "core/backend.h"
#include "bench.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BENCH_ITERATIONS 100
#define BENCH_WARMUP 10

typedef struct {
    double min_throughput;
    double max_throughput;
    double avg_throughput;
    double min_time;
    double max_time;
    double avg_time;
} bench_result;

bench_result benchmark_patcher(const blt_tensor *entropy, const blt_patcher_config *config, size_t max_patches) {
    blt_patch_info *patches = (blt_patch_info *)malloc(max_patches * sizeof(blt_patch_info));
    BLT_REQUIRE(patches != NULL, "Failed to allocate patches buffer");

    double *times = (double *)malloc(BENCH_ITERATIONS * sizeof(double));
    BLT_REQUIRE(times != NULL, "Failed to allocate times buffer");

    blt_timer timer;
    blt_timer_init(&timer);

    for (int i = 0; i < BENCH_WARMUP; ++i) {
        blt_segment_patches(entropy, NULL, patches, max_patches, config);
    }

    for (int i = 0; i < BENCH_ITERATIONS; ++i) {
        blt_timer_start(&timer);
        blt_segment_patches(entropy, NULL, patches, max_patches, config);
        blt_timer_stop(&timer);
        times[i] = blt_timer_elapsed(&timer);
    }

    double total_time = 0.0;
    double min_time = times[0];
    double max_time = times[0];

    for (int i = 0; i < BENCH_ITERATIONS; ++i) {
        total_time += times[i];
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
    }

    double avg_time = total_time / BENCH_ITERATIONS;

    size_t seq_bytes = entropy->numel * sizeof(float);
    double min_throughput = blt_throughput_gb_per_sec(seq_bytes, max_time);
    double max_throughput = blt_throughput_gb_per_sec(seq_bytes, min_time);
    double avg_throughput = blt_throughput_gb_per_sec(seq_bytes, avg_time);

    bench_result result = {.min_throughput = min_throughput,
                           .max_throughput = max_throughput,
                           .avg_throughput = avg_throughput,
                           .min_time = min_time,
                           .max_time = max_time,
                           .avg_time = avg_time};

    free(patches);
    free(times);

    return result;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <entropy_file> [sequence_size]\n", argv[0]);
        fprintf(stderr, "  entropy_file: path to binary FP32 entropy array\n");
        fprintf(stderr, "  sequence_size: sequence length (default: file size / sizeof(float))\n");
        return 1;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open %s\n", argv[1]);
        return 1;
    }

    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size % sizeof(float) != 0) {
        fprintf(stderr, "Error: file size not a multiple of sizeof(float)\n");
        fclose(f);
        return 1;
    }

    size_t seq_len = file_size / sizeof(float);
    if (argc >= 3) {
        size_t requested_len = atoi(argv[2]);
        if (requested_len < seq_len) {
            seq_len = requested_len;
        }
    }

    blt_arena *arena = blt_arena_create(file_size, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "Error: arena creation for patcher tests\n");
        return 0;
    }

    size_t entropy_shape[1] = {seq_len};
    blt_tensor entropy_tensor = blt_tensor_create(arena, entropy_shape, 1, BLT_DTYPE_FP32);
    float *entropy_data = (float *)entropy_tensor.data;

    if (fread(entropy_data, sizeof(float), file_size / sizeof(float), f) != file_size / sizeof(float)) {
        fprintf(stderr, "Error: read failed\n");
        free(entropy_data);
        fclose(f);
        return 1;
    }
    fclose(f);

    size_t max_patches = seq_len;

    blt_patcher_config config_global = {.threshold_global = 1.5f,
                                        .threshold_monotonic = 0.0f,
                                        .max_patch_length = 256,
                                        .rule = BLT_PATCH_RULE_GLOBAL,
                                        .reset_on_newline = false};

    blt_patcher_config config_monotonic = {.threshold_global = 10.0f,
                                           .threshold_monotonic = 0.5f,
                                           .max_patch_length = 256,
                                           .rule = BLT_PATCH_RULE_MONOTONIC,
                                           .reset_on_newline = false};

    blt_patcher_config config_both = {.threshold_global = 1.5f,
                                      .threshold_monotonic = 0.5f,
                                      .max_patch_length = 256,
                                      .rule = BLT_PATCH_RULE_BOTH,
                                      .reset_on_newline = false};

    printf("----------------------------------------\n");
    printf("BLT Patcher Microbenchmark\n");
    printf("----------------------------------------\n");
    printf("Sequence length: %zu bytes (%zu FP32 elements)\n", seq_len * sizeof(float), seq_len);
    printf("Iterations: %d (after %d warmup)\n", BENCH_ITERATIONS, BENCH_WARMUP);
    printf("\n");

    printf("Configuration: GLOBAL THRESHOLD\n");
    bench_result result_global = benchmark_patcher(&entropy_tensor, &config_global, max_patches);
    printf("  Min throughput: %.3f GB/s\n", result_global.min_throughput);
    printf("  Max throughput: %.3f GB/s\n", result_global.max_throughput);
    printf("  Avg throughput: %.3f GB/s\n", result_global.avg_throughput);
    printf("  Min time: %.6f s\n", result_global.min_time);
    printf("  Max time: %.6f s\n", result_global.max_time);
    printf("  Avg time: %.6f s\n", result_global.avg_time);
    printf("\n");

    printf("Configuration: MONOTONIC THRESHOLD\n");
    bench_result result_monotonic = benchmark_patcher(&entropy_tensor, &config_monotonic, max_patches);
    printf("  Min throughput: %.3f GB/s\n", result_monotonic.min_throughput);
    printf("  Max throughput: %.3f GB/s\n", result_monotonic.max_throughput);
    printf("  Avg throughput: %.3f GB/s\n", result_monotonic.avg_throughput);
    printf("  Min time: %.6f s\n", result_monotonic.min_time);
    printf("  Max time: %.6f s\n", result_monotonic.max_time);
    printf("  Avg time: %.6f s\n", result_monotonic.avg_time);
    printf("\n");

    printf("Configuration: BOTH THRESHOLDS\n");
    bench_result result_both = benchmark_patcher(&entropy_tensor, &config_both, max_patches);
    printf("  Min throughput: %.3f GB/s\n", result_both.min_throughput);
    printf("  Max throughput: %.3f GB/s\n", result_both.max_throughput);
    printf("  Avg throughput: %.3f GB/s\n", result_both.avg_throughput);
    printf("  Min time: %.6f s\n", result_both.min_time);
    printf("  Max time: %.6f s\n", result_both.max_time);
    printf("  Avg time: %.6f s\n", result_both.avg_time);
    printf("\n");

    double fastest_throughput = result_global.avg_throughput;
    if (result_monotonic.avg_throughput > fastest_throughput) {
        fastest_throughput = result_monotonic.avg_throughput;
    }
    if (result_both.avg_throughput > fastest_throughput) {
        fastest_throughput = result_both.avg_throughput;
    }

    printf("----------------------------------------\n");
    printf("Summary\n");
    printf("----------------------------------------\n");
    printf("Fastest configuration: %.3f GB/s\n", fastest_throughput);
    if (fastest_throughput > 1.0f) {
        printf("YES - Performance exceeds 1.0 GB/s threshold\n");
    } else {
        printf("NO - Performance below 1.0 GB/s threshold\n");
    }
    printf("\n");

    blt_arena_destroy(arena);

    return 0;
}