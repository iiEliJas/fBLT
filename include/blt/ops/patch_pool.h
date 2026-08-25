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
    BLT_POOL_MEAN,   // default: average byte representations in patch
    BLT_POOL_MAX     // option: per-channel max byte representation in patch
} blt_patch_pool_type;


// Input:     byte_hidden [seq_len, embed_dim], patches array [num_patches], pool_type.
// Output:    out [num_patches, embed_dim] containing pooled patch representations.
// Behavior:  Pools byte representations within each patch boundary using either
//            mean averaging or per-channel maximum.
void blt_patch_pool_forward(const blt_tensor* byte_hidden,
                            const blt_patch_info* patches,
                            size_t num_patches,
                            blt_patch_pool_type pool_type,
                            blt_tensor* out);


// Input:     grad_out [num_patches, embed_dim], byte_hidden [seq_len, embed_dim],
//            patches array, pool_type.
// Output:    grad_byte_hidden [seq_len, embed_dim] (scatter-add target).
// Behavior:  Distributes patch output gradients back to constituent bytes (equally for MEAN,
//            or routed exclusively to argmax positions for MAX).
void blt_patch_pool_backward(const blt_tensor* grad_out,
                             const blt_tensor* byte_hidden,
                             const blt_patch_info* patches,
                             size_t num_patches,
                             blt_patch_pool_type pool_type,
                             blt_tensor* grad_byte_hidden);


// Input:     patches array [num_patches], seq_len (total byte count).
// Output:    query_group_ids_out [num_patches], kv_group_ids_out [seq_len].
// Behavior:  Maps query patches (ID = patch index) and input bytes (ID = parent patch index)
//            to construct group IDs for block-diagonal cross-attention masking.
void blt_patch_build_group_ids(const blt_patch_info* patches,
                               size_t num_patches,
                               size_t seq_len,
                               size_t* query_group_ids_out,
                               size_t* kv_group_ids_out);


// Input:     group_ids_in [n], k (expansion factor).
// Output:    group_ids_out [n * k].
// Behavior:  Expands each input group ID into k consecutive output group IDs.
void blt_patch_expand_group_ids(const size_t* group_ids_in, size_t n, size_t k,
                                size_t* group_ids_out);



#ifdef __cplusplus
}
#endif

#endif // BLT_PATCH_POOL_H