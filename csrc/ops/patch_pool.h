#ifndef BLT_PATCH_POOL_H
#define BLT_PATCH_POOL_H

#ifdef __cplusplus
extern "C" {
#endif

#include "core/backend.h"
#include "core/allocator.h"
#include "ops/mask_builder.h"
#include "models/patcher.h"

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

// Build group IDs for block-diagonal cross-attention masking over the covered
// prefix: query patches get ID = patch index; bytes get ID = parent patch
// index. Does not apply the paper rule -- see blt_patch_decoder_latent_at.
void blt_patch_build_group_ids(const blt_patch_info *patches, size_t num_patches, size_t seq_len,
                               size_t *query_group_ids_out, size_t *kv_group_ids_out);

// Paper-rule decoder latent index for byte position pos (Fast-BLT 3.1.1):
// final byte of patch j -> j; non-final byte of patch j -> (j == 0 ? 0 : j-1);
// position not covered by any patch -> num_patches-1.
size_t blt_patch_decoder_latent_at(const blt_patch_info *patches, size_t num_patches, size_t pos);

// Expand each group ID into k consecutive output IDs.
void blt_patch_expand_group_ids(const size_t *group_ids_in, size_t n, size_t k, size_t *group_ids_out);

#ifdef __cplusplus
}
#endif

#endif
