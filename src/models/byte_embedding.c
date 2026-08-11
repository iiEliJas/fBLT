#include "blt/models/byte_embedding.h"
#include "blt/core/backend.h"


//--------------------------------------------------------------
// Allocates the [256, embed_dim] FP32 embedding table via the arena.
// Caller is responsible for filling the weight->data (random init, or loads pretrained) separately

blt_byte_embedding blt_byte_embedding_create(blt_arena* arena, size_t embed_dim) {
    blt_byte_embedding emb;
    const size_t shape[2] = {256, embed_dim};
    emb.weight = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    emb.embed_dim = embed_dim;
    return emb;
}



//----------------------------------------------------------------
// Forward Pass for Byte Embedding
//
// For position i in [0, seq_len-1]:
//     out[i, :] = weight[bytes_in[i], :]
//
// Shape mapping:
//   bytes_in:    [seq_len]                 (UINT8)
//   emb->weight: [256, embed_dim]          (FP32)
//   out:         [seq_len, embed_dim]      (FP32)
//----------------------------------------------------------------

void blt_byte_embedding_forward(const blt_byte_embedding* emb, const blt_tensor* bytes_in,
                                 blt_tensor* out) {
    // ---- 1. Input Validation ----
    BLT_REQUIRE(bytes_in->dtype == BLT_DTYPE_UINT8, "bytes_in must be a UINT8 tensor");
    BLT_REQUIRE(bytes_in->ndim == 1, "bytes_in must be a 1D tensor of shape [seq_len]");

    BLT_REQUIRE(out->dtype == BLT_DTYPE_FP32, "out must be a FP32 tensor");
    BLT_REQUIRE(out->ndim == 2, "out must be a 2D tensor of shape [seq_len, embed_dim]");
    BLT_REQUIRE(out->shape[0] == bytes_in->shape[0], "out.shape[0] must match bytes_in.shape[0] (seq_len)");
    BLT_REQUIRE(out->shape[1] == emb->embed_dim, "out.shape[1] must match emb->embed_dim");

    const size_t seq_len = bytes_in->shape[0];
    const size_t embed_dim = emb->embed_dim;

    const uint8_t* bytes_data = (const uint8_t*)bytes_in->data;
    const float* weight_data = (const float*)emb->weight.data;
    float* out_data = (float*)out->data;

    // ---- 2. Gather Operation ----
    // Direct lookup: copy embedding row (byte_val) directly to output row i
    for (size_t i = 0; i < seq_len; ++i) {
        uint8_t byte_val = bytes_data[i];
        
        const float* row_in = weight_data + (size_t)byte_val * embed_dim;
        float* row_out = out_data + i * embed_dim;
        
        for (size_t d = 0; d < embed_dim; ++d) {
            row_out[d] = row_in[d];
        }
    }
}



//----------------------------------------------------------------
// Backward Pass for Byte Embedding
//
// Scatter-Add
// For position i in [0, seq_len-1]:
//     grad_weight[bytes_in[i], :] += grad_out[i, :]
//
// Shape mapping:
//   bytes_in:    [seq_len]                 (UINT8)
//   grad_out:    [seq_len, embed_dim]      (FP32)
//   grad_weight: [256, embed_dim]          (FP32, pre-zeroed)
//----------------------------------------------------------------

void blt_byte_embedding_backward(const blt_byte_embedding* emb, const blt_tensor* bytes_in,
                                  const blt_tensor* grad_out, blt_tensor* grad_weight) {
    // ---- 1. Input Validation ----
    BLT_REQUIRE(bytes_in->dtype == BLT_DTYPE_UINT8, "bytes_in must be a UINT8 tensor");
    BLT_REQUIRE(bytes_in->ndim == 1, "bytes_in must be a 1D tensor of shape [seq_len]");

    BLT_REQUIRE(grad_out->dtype == BLT_DTYPE_FP32, "grad_out must be a FP32 tensor");
    BLT_REQUIRE(grad_out->ndim == 2, "grad_out must be a 2D tensor of shape [seq_len, embed_dim]");
    BLT_REQUIRE(grad_out->shape[0] == bytes_in->shape[0], "grad_out.shape[0] must match bytes_in.shape[0] (seq_len)");
    BLT_REQUIRE(grad_out->shape[1] == emb->embed_dim, "grad_out.shape[1] must match emb->embed_dim");

    BLT_REQUIRE(grad_weight->dtype == BLT_DTYPE_FP32, "grad_weight must be a FP32 tensor");
    BLT_REQUIRE(grad_weight->ndim == 2, "grad_weight must be a 2D tensor of shape [256, embed_dim]");
    BLT_REQUIRE(grad_weight->shape[0] == 256, "grad_weight.shape[0] must be 256");
    BLT_REQUIRE(grad_weight->shape[1] == emb->embed_dim, "grad_weight.shape[1] must match emb->embed_dim");

    const size_t seq_len = bytes_in->shape[0];
    const size_t embed_dim = emb->embed_dim;

    const uint8_t* bytes_data = (const uint8_t*)bytes_in->data;
    const float* grad_out_data = (const float*)grad_out->data;
    float* grad_weight_data = (float*)grad_weight->data;

    // ---- 2. Scatter-Add Accumulation ----
    // For each token accumulate grad_out[i, :] into grad_weight[byte_val, :]
    for (size_t i = 0; i < seq_len; ++i) {
        uint8_t byte_val = bytes_data[i];
        
        const float* row_grad = grad_out_data + i * embed_dim;
        float* row_target = grad_weight_data + (size_t)byte_val * embed_dim;
        
        for (size_t d = 0; d < embed_dim; ++d) {
            row_target[d] += row_grad[d];
        }
    }
}