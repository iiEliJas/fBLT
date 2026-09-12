#ifndef BLT_PATCH_POOL_H
#define BLT_PATCH_POOL_H

#ifdef __cplusplus
extern "C" {
#endif

#include "blt/core/backend.h"
#include "blt/core/allocator.h"
#include "blt/ops/mask_builder.h"
#include "blt/models/patcher.h"

typedef enum {
    BLT_POOL_MEAN, // average byte representations in patch
    BLT_POOL_MAX   // per-channel max byte representation in patch
} blt_patch_pool_type;

// Pools byte representations within each patch boundary (mean or per-channel max).
// byte_hidden [seq_len, embed_dim] -> out [num_patches, embed_dim].
void blt_patch_pool_forward(const blt_tensor *byte_hidden, const blt_patch_info *patches, size_t num_patches,
                            blt_patch_pool_type pool_type, blt_tensor *out);

// Scatter-add gradients back to bytes: MEAN distributes equally, MAX routes to argmax.
// grad_out [num_patches, embed_dim] -> grad_byte_hidden [seq_len, embed_dim].
void blt_patch_pool_backward(const blt_tensor *grad_out, const blt_tensor *byte_hidden, const blt_patch_info *patches,
                             size_t num_patches, blt_patch_pool_type pool_type, blt_tensor *grad_byte_hidden);

// Build group IDs for block-diagonal cross-attention masking.
// Query patches get ID = patch index; bytes get ID = parent patch index.
void blt_patch_build_group_ids(const blt_patch_info *patches, size_t num_patches, size_t seq_len,
                               size_t *query_group_ids_out, size_t *kv_group_ids_out);

// Expand each group ID into k consecutive output IDs.
void blt_patch_expand_group_ids(const size_t *group_ids_in, size_t n, size_t k, size_t *group_ids_out);

#ifdef __cplusplus
}
#endif

#endif
