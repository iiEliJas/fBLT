#include "blt/core/backend.h"
#include "blt/models/model.h"


//----------------------------------------------------------------------
// Doc boundary remapping
//
static void map_byte_boundaries_to_patch_boundaries(
    const size_t* byte_doc_boundaries, size_t num_docs,
    const blt_patch_info* patches, size_t num_patches,
    size_t* patch_doc_boundaries_out
) {
    size_t patch_idx = 0;
    for (size_t d = 0; d < num_docs; d++) {
        size_t byte_boundary = byte_doc_boundaries[d];
        while (patch_idx < num_patches && patches[patch_idx].start_idx < byte_boundary) {
            patch_idx++;
        }
        BLT_REQUIRE(patch_idx < num_patches,
            "map_byte_boundaries_to_patch_boundaries: doc boundary does not align to a patch start");
        BLT_REQUIRE(patches[patch_idx].start_idx == byte_boundary,
            "map_byte_boundaries_to_patch_boundaries: doc boundary is not a patch boundary");
        patch_doc_boundaries_out[d] = patch_idx;
    }
}



//----------------------------------------------------------------------
// Create
//
blt_model* blt_model_create(blt_arena* arena, const blt_model_config* config) {
    BLT_REQUIRE(arena != NULL && config != NULL,
        "blt_model_create: arena and config cannot be NULL");
    BLT_REQUIRE(
        config->encoder_config.embed_dim == config->global_config.embed_dim &&
        config->global_config.embed_dim == config->decoder_config.embed_dim,
        "blt_model_create: encoder/global/decoder embed_dim must match (shared hidden width)");

    blt_model* model = (blt_model*)blt_arena_alloc(arena, sizeof(blt_model), sizeof(void*));
    model->config = *config;

    model->encoder = blt_local_encoder_create(arena, &config->encoder_config);
    model->global  = blt_global_transformer_create(arena, &config->global_config);
    model->decoder = blt_local_decoder_create(arena, &config->decoder_config);

    return model;
}



blt_model_grad* blt_model_grad_create(blt_arena* arena, const blt_model* model) {
    BLT_REQUIRE(arena != NULL && model != NULL,
        "blt_model_grad_create: arena and model cannot be NULL");

    blt_model_grad* grad = (blt_model_grad*)blt_arena_alloc(arena, sizeof(blt_model_grad), sizeof(void*));

    grad->encoder_grad = blt_local_encoder_grad_create(arena, model->encoder);
    grad->global_grad  = blt_global_transformer_grad_create(arena, model->global);
    grad->decoder_grad = blt_local_decoder_grad_create(arena, model->decoder);

    return grad;
}



//----------------------------------------------------------------------
// Forward path
//
// Pipeline:
//   patch_out, byte_hidden_out = Encoder(bytes_in, patches)
//   patch_doc_boundaries       = byte doc_boundaries -> patch indices
//   O                          = Global(patch_out, patch_doc_boundaries)
//   logits_out, loss_out       = Decoder(byte_hidden_out, O, bytes_in)
//
void blt_model_forward(
    const blt_model* model,
    const blt_tensor* bytes_in,
    const blt_patch_info* patches,
    size_t num_patches,
    const size_t* doc_boundaries,
    size_t num_docs,
    blt_tensor* logits_out,
    blt_tensor* loss_out,
    blt_arena* arena
) {
    BLT_REQUIRE(model != NULL && bytes_in != NULL && patches != NULL &&
                logits_out != NULL && arena != NULL,
        "blt_model_forward: arguments cannot be NULL");
    BLT_REQUIRE(bytes_in->ndim == 1 && bytes_in->dtype == BLT_DTYPE_UINT8,
        "blt_model_forward: bytes_in must be a 1D UINT8 tensor [seq_len]");
    BLT_REQUIRE(num_docs == 0 || doc_boundaries != NULL,
        "blt_model_forward: doc_boundaries cannot be NULL when num_docs > 0");
    BLT_REQUIRE(num_patches >= 1, "blt_model_forward: num_patches must be >= 1");

    size_t seq_len = bytes_in->shape[0];
    size_t embed_dim = model->config.encoder_config.embed_dim;
    size_t patch_shape[2] = {num_patches, embed_dim};
    size_t byte_shape[2] = {seq_len, embed_dim};

    // -----------------------------------------------------------------
    // 1. Local encoder: bytes + patches -> P_final, h_final
    // -----------------------------------------------------------------
    blt_tensor patch_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    blt_tensor byte_hidden_out = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);

    blt_local_encoder_forward(model->encoder, bytes_in, patches, num_patches,
        doc_boundaries, num_docs, &patch_out, &byte_hidden_out, arena);

    // -----------------------------------------------------------------
    // 2. Remap doc boundaries from byte indices to patch indices
    // -----------------------------------------------------------------
    size_t* patch_doc_boundaries = NULL;
    if (num_docs > 0) {
        patch_doc_boundaries = (size_t*)blt_arena_alloc(
            arena, num_docs * sizeof(size_t), sizeof(size_t));
        map_byte_boundaries_to_patch_boundaries(
            doc_boundaries, num_docs, patches, num_patches, patch_doc_boundaries);
    }

    // -----------------------------------------------------------------
    // 3. Global transformer: P_final -> O
    // -----------------------------------------------------------------
    blt_tensor global_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    blt_global_transformer_forward(model->global, &patch_out,
        patch_doc_boundaries, num_docs, &global_out, arena);

    // -----------------------------------------------------------------
    // 4. Local decoder: h_final + O -> logits, loss
    // -----------------------------------------------------------------
    blt_local_decoder_forward(model->decoder, &byte_hidden_out, &global_out,
        patches, num_patches, bytes_in, doc_boundaries, num_docs,
        logits_out, loss_out, arena);
}



//----------------------------------------------------------------------
// Backward path
//
// Mirrors forward call order in reverse. Forward intermediates
// (patch_out, byte_hidden_out, global_out) are being recomputed
//
void blt_model_backward(
    const blt_model* model,
    const blt_tensor* bytes_in,
    const blt_patch_info* patches,
    size_t num_patches,
    const size_t* doc_boundaries,
    size_t num_docs,
    blt_model_grad* grad,
    blt_arena* arena
) {
    BLT_REQUIRE(model != NULL && bytes_in != NULL && patches != NULL &&
                grad != NULL && arena != NULL,
        "blt_model_backward: arguments cannot be NULL");
    BLT_REQUIRE(bytes_in->ndim == 1 && bytes_in->dtype == BLT_DTYPE_UINT8,
        "blt_model_backward: bytes_in must be a 1D UINT8 tensor [seq_len]");
    BLT_REQUIRE(num_docs == 0 || doc_boundaries != NULL,
        "blt_model_backward: doc_boundaries cannot be NULL when num_docs > 0");
    BLT_REQUIRE(num_patches >= 1, "blt_model_backward: num_patches must be >= 1");

    size_t seq_len = bytes_in->shape[0];
    size_t embed_dim = model->config.encoder_config.embed_dim;
    size_t patch_shape[2] = {num_patches, embed_dim};
    size_t byte_shape[2] = {seq_len, embed_dim};

    // ----------------
    // Recompute the forward intermediates needed by the submodule
    // backward calls.
    blt_tensor patch_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    blt_tensor byte_hidden_out = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_local_encoder_forward(model->encoder, bytes_in, patches, num_patches,
        doc_boundaries, num_docs, &patch_out, &byte_hidden_out, arena);

    size_t* patch_doc_boundaries = NULL;
    if (num_docs > 0) {
        patch_doc_boundaries = (size_t*)blt_arena_alloc(
            arena, num_docs * sizeof(size_t), sizeof(size_t));
        map_byte_boundaries_to_patch_boundaries(
            doc_boundaries, num_docs, patches, num_patches, patch_doc_boundaries);
    }

    blt_tensor global_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    blt_global_transformer_forward(model->global, &patch_out,
        patch_doc_boundaries, num_docs, &global_out, arena);

    // ----------------
    // 1. Local decoder backward: dL/d(h_final), dL/d(O)
    blt_tensor grad_byte_hidden = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_tensor grad_global_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);

    blt_local_decoder_backward(model->decoder, &byte_hidden_out, &global_out,
        patches, num_patches, bytes_in, doc_boundaries, num_docs,
        &grad_byte_hidden, &grad_global_out, grad->decoder_grad, arena);

    // ----------------
    // 2. Global transformer backward: dL/d(P_final)
    blt_tensor grad_patch_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    blt_global_transformer_backward(model->global, &patch_out,
        patch_doc_boundaries, num_docs, &grad_global_out, &grad_patch_out,
        grad->global_grad, arena);

    // ----------------
    // 3. Local encoder backward: consumes grad_patch_out and grad_byte_hidden
    blt_local_encoder_backward(model->encoder, bytes_in, patches, num_patches,
        doc_boundaries, num_docs, &grad_patch_out, &grad_byte_hidden,
        grad->encoder_grad, arena);
}