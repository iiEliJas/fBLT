#ifndef BLT_INFER_ROPE_GATHER_H
#define BLT_INFER_ROPE_GATHER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "core/tensor.h"
#include "core/allocator.h"

// Gathers RoPE cos/sin rows for arbitrary position ids.
//
// The decoder applies RoPE by row index == position, which is correct for
// contiguous drafting (BLT-S) but wrong for BLT-D block rows whose row
// index differs from their original sequence position. This util builds
// position-correct cos/sin views: out row i is a copy of cache row
// positions[i].
//
// Input:  cos_cache / sin_cache [max_seq_len, head_dim/2] FP32 (as produced
//         by blt_rope_precompute), positions[n] with each entry < max_seq_len.
// Output: cos_out / sin_out created as [n, head_dim/2] FP32 tensors in arena.
void blt_rope_position_gather(const blt_tensor *cos_cache, const blt_tensor *sin_cache, const size_t *positions,
                              size_t n, blt_tensor *cos_out, blt_tensor *sin_out, blt_arena *arena);

#ifdef __cplusplus
}
#endif

#endif // BLT_INFER_ROPE_GATHER_H
