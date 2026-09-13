#include "infer/rope_gather.h"
#include "core/backend.h"
#include "ops/gather_scatter.h"

void blt_rope_position_gather(const blt_tensor *cos_cache, const blt_tensor *sin_cache, const size_t *positions,
                              size_t n, blt_tensor *cos_out, blt_tensor *sin_out, blt_arena *arena) {
    BLT_REQUIRE(cos_cache != NULL && sin_cache != NULL && positions != NULL && cos_out != NULL && sin_out != NULL &&
                    arena != NULL,
                "blt_rope_position_gather: arguments cannot be NULL");
    BLT_REQUIRE(n >= 1, "blt_rope_position_gather: n must be >= 1");
    BLT_REQUIRE(cos_cache->ndim == 2 && cos_cache->dtype == BLT_DTYPE_FP32,
                "blt_rope_position_gather: cos_cache must be a 2D FP32 tensor");
    BLT_REQUIRE(sin_cache->ndim == 2 && sin_cache->dtype == BLT_DTYPE_FP32,
                "blt_rope_position_gather: sin_cache must be a 2D FP32 tensor");

    const size_t max_seq_len = cos_cache->shape[0];
    const size_t half = cos_cache->shape[1];
    BLT_REQUIRE(sin_cache->shape[0] == max_seq_len && sin_cache->shape[1] == half,
                "blt_rope_position_gather: cos/sin cache shape mismatch");

    for (size_t i = 0; i < n; i++) {
        BLT_REQUIRE(positions[i] < max_seq_len,
                    "blt_rope_position_gather: positions[%zu]=%zu out of cache range [0, %zu)", i, positions[i],
                    max_seq_len);
    }

    // `positions` is a host array; the gather op handles either memory space
    // for the caches and stages the indices internally when needed.
    size_t out_shape[2] = {n, half};
    *cos_out = blt_tensor_create(arena, out_shape, 2, BLT_DTYPE_FP32);
    *sin_out = blt_tensor_create(arena, out_shape, 2, BLT_DTYPE_FP32);

    blt_rows_gather(cos_cache, positions, cos_out);
    blt_rows_gather(sin_cache, positions, sin_out);
}
