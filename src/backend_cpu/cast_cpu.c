#include "blt/core/backend.h"
#include "blt/ops/cast.h"

#include <string.h>
#include <stdint.h>

// Software bf16 round-trip: fp32 -> bf16 (truncate to 16 bits with
// round-to-nearest-even) -> fp32.  Deterministic and bit-exact on all
// platforms, which makes it the CPU ground truth for CUDA parity tests.

static uint16_t fp32_to_bf16_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    // Round-to-nearest-even on the low 16 bits, then truncate.
    uint32_t rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    return (uint16_t)((bits + rounding_bias) >> 16);
}

static float bf16_bits_to_fp32(uint16_t h) {
    // bf16 and fp32 share the same exponent bias (127) and the same
    // exponent width (8 bits). The only difference is the mantissa
    // width (7 vs 23 bits). Converting bf16 to fp32 is therefore a
    // simple left-shift by 16: the sign, exponent, and mantissa bits
    // land in exactly the right fp32 positions.
    uint32_t bits = (uint32_t)h << 16;
    float result;
    memcpy(&result, &bits, 4);
    return result;
}


void blt_cast_cpu(const blt_tensor* in, blt_tensor* out) {
    BLT_REQUIRE(in != NULL && out != NULL, "blt_cast: in and out must not be NULL");
    BLT_REQUIRE(in->numel == out->numel, "blt_cast: in and out must have the same numel");

    // fp32 -> bf16
    if (in->dtype == BLT_DTYPE_FP32 && out->dtype == BLT_DTYPE_BF16) {
        const float* src = (const float*)in->data;
        uint16_t* dst = (uint16_t*)out->data;
        for (size_t i = 0; i < in->numel; ++i) {
            dst[i] = fp32_to_bf16_bits(src[i]);
        }
        return;
    }

    // bf16 -> fp32
    if (in->dtype == BLT_DTYPE_BF16 && out->dtype == BLT_DTYPE_FP32) {
        const uint16_t* src = (const uint16_t*)in->data;
        float* dst = (float*)out->data;
        for (size_t i = 0; i < in->numel; ++i) {
            dst[i] = bf16_bits_to_fp32(src[i]);
        }
        return;
    }

    BLT_FATAL("blt_cast: unsupported conversion from dtype %d to %d", in->dtype, out->dtype);
}
