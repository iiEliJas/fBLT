#include "blt/models/byte_embedding.h"
#include "blt/core/backend.h"
#include "blt/ops/gather_scatter.h"

// Allocates [vocab_size, embed_dim] FP32 table via arena.
// Caller fills weight->data separately (random init or pretrained).
blt_byte_embedding blt_byte_embedding_create(blt_arena *arena, size_t vocab_size, size_t embed_dim) {
    blt_byte_embedding emb;
    const size_t shape[2] = {vocab_size, embed_dim};
    emb.vocab_size = vocab_size;
    emb.weight = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    emb.embed_dim = embed_dim;
    return emb;
}

void blt_byte_embedding_forward(const blt_byte_embedding *emb, const blt_tensor *bytes_in, blt_tensor *out) {
    BLT_REQUIRE(bytes_in->dtype == BLT_DTYPE_UINT8, "bytes_in must be a UINT8 tensor");
    BLT_REQUIRE(bytes_in->ndim == 1, "bytes_in must be a 1D tensor of shape [seq_len]");

    BLT_REQUIRE(out->dtype == BLT_DTYPE_FP32, "out must be a FP32 tensor");
    BLT_REQUIRE(out->ndim == 2, "out must be a 2D tensor of shape [seq_len, embed_dim]");
    BLT_REQUIRE(out->shape[0] == bytes_in->shape[0], "out.shape[0] must match bytes_in.shape[0] (seq_len)");
    BLT_REQUIRE(out->shape[1] == emb->embed_dim, "out.shape[1] must match emb->embed_dim");

    const size_t seq_len = bytes_in->shape[0];
    const size_t embed_dim = emb->embed_dim;
    (void)seq_len;
    (void)embed_dim;

    // Ids may live on either backend; the lookup consumes them on the
    // host side of the dispatched op, so device inputs are staged first.
    uint8_t *stage = NULL;
    const uint8_t *ids = bytes_in->data;
    if (bytes_in->backend != BLT_BACKEND_CPU) {
        stage = (uint8_t *)malloc(bytes_in->numel);
        BLT_REQUIRE(stage != NULL, "blt_byte_embedding_forward: staging alloc failed");
        blt_tensor_download(bytes_in, stage, bytes_in->numel);
        ids = stage;
    }
    blt_embedding_lookup(&emb->weight, ids, out);
    free(stage);
}

// Scatter-add: grad_weight[bytes_in[i], :] += grad_out[i, :]
// grad_weight must be pre-zeroed by caller.
void blt_byte_embedding_backward(const blt_byte_embedding *emb, const blt_tensor *bytes_in, const blt_tensor *grad_out,
                                 blt_tensor *grad_weight) {
    BLT_REQUIRE(bytes_in->dtype == BLT_DTYPE_UINT8, "bytes_in must be a UINT8 tensor");
    BLT_REQUIRE(bytes_in->ndim == 1, "bytes_in must be a 1D tensor of shape [seq_len]");

    BLT_REQUIRE(grad_out->dtype == BLT_DTYPE_FP32, "grad_out must be a FP32 tensor");
    BLT_REQUIRE(grad_out->ndim == 2, "grad_out must be a 2D tensor of shape [seq_len, embed_dim]");
    BLT_REQUIRE(grad_out->shape[0] == bytes_in->shape[0], "grad_out.shape[0] must match bytes_in.shape[0] (seq_len)");
    BLT_REQUIRE(grad_out->shape[1] == emb->embed_dim, "grad_out.shape[1] must match emb->embed_dim");

    BLT_REQUIRE(grad_weight->dtype == BLT_DTYPE_FP32, "grad_weight must be a FP32 tensor");
    BLT_REQUIRE(grad_weight->ndim == 2, "grad_weight must be a 2D tensor of shape [256, embed_dim]");
    BLT_REQUIRE(grad_weight->shape[0] == emb->vocab_size, "grad_weight.shape[0] must be vocab_size");
    BLT_REQUIRE(grad_weight->shape[1] == emb->embed_dim, "grad_weight.shape[1] must match emb->embed_dim");

    const size_t seq_len = bytes_in->shape[0];
    (void)seq_len;

    uint8_t *stage = NULL;
    const uint8_t *ids = bytes_in->data;
    if (bytes_in->backend != BLT_BACKEND_CPU) {
        stage = (uint8_t *)malloc(bytes_in->numel);
        BLT_REQUIRE(stage != NULL, "blt_byte_embedding_backward: staging alloc failed");
        blt_tensor_download(bytes_in, stage, bytes_in->numel);
        ids = stage;
    }
    blt_embedding_scatter_add(grad_weight, ids, grad_out);
    free(stage);
}
