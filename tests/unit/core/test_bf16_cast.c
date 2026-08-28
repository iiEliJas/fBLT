#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "blt/core/allocator.h"
#include "blt/core/backend.h"
#include "blt/ops/cast.h"
#include "blt/ops/matmul.h"

#include "test_helpers.h"
#include "test_suite.h"


// Reference bf16 round-trip (same algorithm as cast_cpu.c, independent impl)
static uint16_t ref_fp32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    return (uint16_t)((bits + rounding_bias) >> 16);
}

static float ref_bf16_to_fp32(uint16_t h) {
    // bf16 and fp32 share exponent bias (127) and width (8 bits).
    // Conversion is a simple left-shift by 16.
    uint32_t bits = (uint32_t)h << 16;
    float result;
    memcpy(&result, &bits, 4);
    return result;
}


int run_bf16_cast_tests(void) {
    blt_arena* arena = blt_arena_create(16384, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);

    // ---------------------------------------------------------------
    // Test 1: fp32 -> bf16 -> fp32 round-trip preserves known values
    // ---------------------------------------------------------------
    {
        size_t n = 8;
        size_t shape[2] = {1, n};
        blt_tensor fp32_in = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
        blt_tensor bf16_tmp = blt_tensor_create(arena, shape, 2, BLT_DTYPE_BF16);
        blt_tensor fp32_out = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);

        float* in = (float*)fp32_in.data;
        // Test values: exact bf16-representable, small, large, denormal-adjacent
        in[0] = 1.0f;
        in[1] = 0.5f;
        in[2] = 3.14159f;
        in[3] = 1000.0f;
        in[4] = 0.0f;
        in[5] = -1.0f;
        in[6] = 65504.0f;   // max bf16 finite value
        in[7] = 1e-7f;       // subnormal range for bf16

        blt_cast(&fp32_in, &bf16_tmp);
        blt_cast(&bf16_tmp, &fp32_out);

        float* out = (float*)fp32_out.data;
        for (size_t i = 0; i < n; ++i) {
            uint16_t expected_bf16 = ref_fp32_to_bf16(in[i]);
            float expected_fp32 = ref_bf16_to_fp32(expected_bf16);
            TEST_ASSERT(out[i] == expected_fp32);
        }
    }

    // ---------------------------------------------------------------
    // Test 2: bf16 -> fp32 -> bf16 round-trip preserves known bits
    // ---------------------------------------------------------------
    {
        size_t n = 6;
        size_t bf16_shape[2] = {1, n};
        size_t fp32_shape[2] = {1, n};
        blt_tensor bf16_in = blt_tensor_create(arena, bf16_shape, 2, BLT_DTYPE_BF16);
        blt_tensor fp32_tmp = blt_tensor_create(arena, fp32_shape, 2, BLT_DTYPE_FP32);
        blt_tensor bf16_out = blt_tensor_create(arena, bf16_shape, 2, BLT_DTYPE_BF16);

        uint16_t* bf16_in_data = (uint16_t*)bf16_in.data;
        // Hand-picked bf16 bit patterns
        bf16_in_data[0] = 0x3F80;  // 1.0
        bf16_in_data[1] = 0x4000;  // 2.0
        bf16_in_data[2] = 0x0000;  // 0.0
        bf16_in_data[3] = 0xBF80;  // -1.0
        bf16_in_data[4] = 0x7BFF;  // max normal bf16 (65504.0)
        bf16_in_data[5] = 0x3C00;  // 0.0078125 (smallest power-of-2 normal)

        blt_cast(&bf16_in, &fp32_tmp);
        blt_cast(&fp32_tmp, &bf16_out);

        uint16_t* bf16_out_data = (uint16_t*)bf16_out.data;
        for (size_t i = 0; i < n; ++i) {
            if (bf16_out_data[i] != bf16_in_data[i]) {
                fprintf(stderr, "  [bf16 round-trip] idx=%zu: in=0x%04X out=0x%04X fp32_intermediate=%.10g\n",
                        i, bf16_in_data[i], bf16_out_data[i], ((float*)fp32_tmp.data)[i]);
                blt_arena_destroy(arena);
                return 0;
            }
        }
    }

    // ---------------------------------------------------------------
    // Test 3: bf16 matmul (bf16 inputs -> fp32 output) matches fp32 matmul
    //          within bf16 precision tolerance
    // ---------------------------------------------------------------
    {
        size_t m = 4, k = 3, n = 2;
        size_t a_shape[2] = {m, k};
        size_t b_shape[2] = {k, n};
        size_t out_shape[2] = {m, n};

        // fp32 path (ground truth)
        blt_tensor a_fp32 = blt_tensor_create(arena, a_shape, 2, BLT_DTYPE_FP32);
        blt_tensor b_fp32 = blt_tensor_create(arena, b_shape, 2, BLT_DTYPE_FP32);
        blt_tensor out_fp32 = blt_tensor_create(arena, out_shape, 2, BLT_DTYPE_FP32);

        float* af = (float*)a_fp32.data;
        float* bf = (float*)b_fp32.data;
        af[0]=1.0f; af[1]=2.0f; af[2]=3.0f;
        af[3]=4.0f; af[4]=5.0f; af[5]=6.0f;
        af[6]=7.0f; af[7]=8.0f; af[8]=9.0f;
        af[9]=10.0f; af[10]=11.0f; af[11]=12.0f;
        bf[0]=0.5f; bf[1]=1.0f;
        bf[2]=1.5f; bf[3]=2.0f;
        bf[4]=2.5f; bf[5]=3.0f;

        blt_matmul(&a_fp32, &b_fp32, &out_fp32);

        // bf16 path
        blt_tensor a_bf16 = blt_tensor_create(arena, a_shape, 2, BLT_DTYPE_BF16);
        blt_tensor b_bf16 = blt_tensor_create(arena, b_shape, 2, BLT_DTYPE_BF16);
        blt_tensor out_bf16 = blt_tensor_create(arena, out_shape, 2, BLT_DTYPE_FP32);

        blt_cast(&a_fp32, &a_bf16);
        blt_cast(&b_fp32, &b_bf16);
        blt_matmul(&a_bf16, &b_bf16, &out_bf16);

        // bf16 matmul result should be close to fp32 matmul result
        // (bf16 has ~3 decimal digits of mantissa precision)
        float* ref = (float*)out_fp32.data;
        float* got = (float*)out_bf16.data;
        for (size_t i = 0; i < m * n; ++i) {
            float diff = fabsf(ref[i] - got[i]);
            float tol = fabsf(ref[i]) * 0.01f + 1e-3f;
            if (diff > tol) {
                fprintf(stderr, "  [FAIL] bf16 matmul: out[%zu] ref=%.6f got=%.6f diff=%.6f tol=%.6f\n",
                        i, ref[i], got[i], diff, tol);
                blt_arena_destroy(arena);
                return 0;
            }
        }
    }

    blt_arena_destroy(arena);
    return 1;
}
