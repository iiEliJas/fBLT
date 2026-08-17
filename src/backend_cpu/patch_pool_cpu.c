#include "blt/ops/patch_pool.h"



void blt_patch_pool_forward(const blt_tensor* byte_hidden, const blt_patch_info* patches, size_t num_patches,
                            blt_patch_pool_type pool_type, blt_tensor* out){

    BLT_REQUIRE(byte_hidden != NULL && out != NULL, "patch_pool: NULL tensor");
    BLT_REQUIRE(patches != NULL || num_patches == 0, "patch_pool: NULL patches");
    BLT_REQUIRE(pool_type == BLT_POOL_MEAN || pool_type == BLT_POOL_MAX,
                "patch_pool: unknown pool_type");
    blt_check_nd_fp32(byte_hidden, 2, (size_t[]){0, 0},
                      "patch_pool: byte_hidden must be 2D FP32");

    const size_t seq_len   = byte_hidden->shape[0];
    const size_t embed_dim = byte_hidden->shape[1];
    blt_check_nd_fp32(out, 2, (size_t[]){num_patches, embed_dim},
                      "patch_pool: out must be [num_patches, embed_dim] FP32");

    const float* h = (const float*)byte_hidden->data;
    float* o = (float*)out->data;

    for (size_t j = 0; j < num_patches; j++) {
        const size_t start = patches[j].start_idx;
        const size_t len   = patches[j].length;
        BLT_REQUIRE(len >= 1, "patch_pool: empty patch");
        BLT_REQUIRE(start + len <= seq_len, "patch_pool: patch out of range");

        float* dst = o + j * embed_dim;
        const float* src0 = h + start * embed_dim;

        if (pool_type == BLT_POOL_MEAN) {
            for (size_t d = 0; d < embed_dim; d++) dst[d] = 0.0f;
            for (size_t i = 0; i < len; i++) {
                const float* src = src0 + i * embed_dim;
                for (size_t d = 0; d < embed_dim; d++) dst[d] += src[d];
            }
            const float inv_len = 1.0f / (float)len;
            for (size_t d = 0; d < embed_dim; d++) dst[d] *= inv_len;
        } else // BLT_POOL_MAX
        { 
            memcpy(dst, src0, embed_dim * sizeof(float));
            for (size_t i = 1; i < len; i++) {
                const float* src = src0 + i * embed_dim;
                for (size_t d = 0; d < embed_dim; d++)
                    if (src[d] > dst[d]) dst[d] = src[d];
            }
        }
    }
}



void blt_patch_pool_backward(const blt_tensor* grad_out, const blt_tensor* byte_hidden, const blt_patch_info* patches, size_t num_patches,
                             blt_patch_pool_type pool_type, blt_tensor* grad_byte_hidden){

    BLT_REQUIRE(grad_out != NULL && byte_hidden != NULL &&
                grad_byte_hidden != NULL, "patch_pool_bw: NULL tensor");
    BLT_REQUIRE(patches != NULL || num_patches == 0, "patch_pool_bw: NULL patches");
    BLT_REQUIRE(pool_type == BLT_POOL_MEAN || pool_type == BLT_POOL_MAX,
                "patch_pool_bw: unknown pool_type");
    blt_check_nd_fp32(byte_hidden, 2, (size_t[]){0, 0},
                      "patch_pool_bw: byte_hidden must be 2D FP32");

    const size_t seq_len   = byte_hidden->shape[0];
    const size_t embed_dim = byte_hidden->shape[1];
    blt_check_nd_fp32(grad_out, 2, (size_t[]){num_patches, embed_dim},
                      "patch_pool_bw: grad_out must be [num_patches, embed_dim]");
    blt_check_nd_fp32(grad_byte_hidden, 2, (size_t[]){seq_len, embed_dim},
                      "patch_pool_bw: grad_byte_hidden must be [seq_len, embed_dim]");

    const float* g  = (const float*)grad_out->data;
    const float* h  = (const float*)byte_hidden->data;
    float* gh = (float*)grad_byte_hidden->data;     // zero init by caller

    for (size_t j = 0; j < num_patches; j++) {
        const size_t start = patches[j].start_idx;
        const size_t len   = patches[j].length;
        BLT_REQUIRE(len >= 1, "patch_pool_bw: empty patch");
        BLT_REQUIRE(start + len <= seq_len, "patch_pool_bw: patch out of range");

        const float* gj = g + j * embed_dim;

        if (pool_type == BLT_POOL_MEAN) {
            const float inv_len = 1.0f / (float)len;
            for (size_t i = 0; i < len; i++) {
                float* dst = gh + (start + i) * embed_dim;
                for (size_t d = 0; d < embed_dim; d++)
                    dst[d] += gj[d] * inv_len;
            }
        } else // BLT_POOL_MAX - recompute argmax per channel
        { 
            for (size_t d = 0; d < embed_dim; d++) {
                size_t argmax = start;
                float best = h[start * embed_dim + d];
                for (size_t i = 1; i < len; i++) {
                    const float v = h[(start + i) * embed_dim + d];
                    if (v > best) { best = v; argmax = start + i; }
                }
                gh[argmax * embed_dim + d] += gj[d];
            }
        }
    }
}



void blt_patch_build_group_ids(const blt_patch_info* patches, size_t num_patches, size_t seq_len,
                               size_t* query_group_ids_out, size_t* kv_group_ids_out) {
    BLT_REQUIRE(patches != NULL || num_patches == 0,
                "patch_group_ids: NULL patches");
    BLT_REQUIRE(query_group_ids_out != NULL && kv_group_ids_out != NULL,
                "patch_group_ids: NULL output array");

    for (size_t j = 0; j < num_patches; j++)
        query_group_ids_out[j] = j;

    size_t pos = 0;
    for (size_t j = 0; j < num_patches; j++) {
        BLT_REQUIRE(patches[j].start_idx == pos,
                    "patch_group_ids: patches must tile [0, seq_len) contiguously");
        BLT_REQUIRE(patches[j].length >= 1, "patch_group_ids: empty patch");
        for (size_t i = 0; i < patches[j].length; i++)
            kv_group_ids_out[pos++] = j;
    }
    BLT_REQUIRE(pos == seq_len,
                "patch_group_ids: patches do not cover [0, seq_len) exactly");
}
