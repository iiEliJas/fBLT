#ifndef BLT_PATCH_POOL_H
#define BLT_PATCH_POOL_H

#include "blt/core/backend.h"
#include "blt/core/allocator.h"
#include "blt/ops/mask_builder.h"
#include "blt/models/patcher.h"

typedef enum {
    BLT_POOL_MEAN,   // default: average the byte reps in the patch
    BLT_POOL_MAX     // option (paper Table 7 init strategies)
} blt_patch_pool_type;



/*
 * blt_patch_pool_forward(byte_hidden, patches, num_patches, pool_type, out, arena)
 *   Input:  byte_hidden [seq_len, embed_dim] FP32,
 *           patches     [num_patches] from blt_segment_patches,
 *           pool_type   BLT_POOL_MEAN or BLT_POOL_MAX.
 *   Output: out [num_patches, embed_dim] FP32 — row j is the pooled rep of
 *           the bytes in patch j (mean or per-channel max).
 *   Notes:  every patch must satisfy length >= 1 and start_idx + length <=
 *           seq_len (checked). arena is unused (no scratch needed); kept in
 *           the signature for API symmetry.
 */
void blt_patch_pool_forward(const blt_tensor* byte_hidden, const blt_patch_info* patches, size_t num_patches,
                            blt_patch_pool_type pool_type, blt_tensor* out);



/*
 * blt_patch_pool_backward(grad_out, byte_hidden, patches, num_patches,
 *                         pool_type, grad_byte_hidden, arena)
 *   Input:  grad_out        [num_patches, embed_dim] FP32,
 *           byte_hidden     [seq_len, embed_dim] FP32 — needed to recompute
 *                           MAX's per-channel argmax (recompute-instead-of-
 *                           cache, same pattern as attention/entropy_lm
 *                           backward).
 *   Output: grad_byte_hidden [seq_len, embed_dim] FP32 — SCATTER-ADD target.
 *           MUST be zero-initialized by the caller before accumulating.
 *   Behavior:
 *     MEAN: grad_out[j] / patch.length is scattered to every byte in patch j.
 *     MAX:  the argmax per channel is recomputed from byte_hidden; the full
 *           gradient for that channel is routed to the argmax byte only
 *           (first max wins on ties, matching forward's strict-> scan).
 */
void blt_patch_pool_backward(const blt_tensor* grad_out, const blt_tensor* byte_hidden, const blt_patch_info* patches,
                             size_t num_patches, blt_patch_pool_type pool_type, blt_tensor* grad_byte_hidden);



/*
 * blt_patch_build_group_ids(patches, num_patches, seq_len,
 *                           query_group_ids_out, kv_group_ids_out)
 *   Input:  patches [num_patches], seq_len = total byte count.
 *   Output: query_group_ids_out [num_patches] - query j's group = j.
 *           kv_group_ids_out      [seq_len]     - byte i's group = id of the
 *                                                 patch that contains it.
 *   Both output arrays are caller-allocated.
 *   Use with blt_mask_config for the cross-attention block-diagonal mask:
 *       .query_group_ids          = query_group_ids_out,
 *       .kv_group_ids             = kv_group_ids_out,
 *       .bidirectional_within_group = true,   (full visibility in a patch)
 *       .is_causal                = false,
 *   Requires the patches to tile [0, seq_len) contiguously (as produced by
 *   blt_segment_patches); aborts via BLT_REQUIRE otherwise - an off-by-one
 *   here silently corrupts every patch downstream.
 */
void blt_patch_build_group_ids(const blt_patch_info* patches, size_t num_patches, size_t seq_len,
                                size_t* query_group_ids_out, size_t* kv_group_ids_out);




#endif // BLT_PATCH_POOL_H