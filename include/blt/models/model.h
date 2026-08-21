// blt/models/blt_model.h
#ifndef BLT_MODELS_BLT_MODEL_H
#define BLT_MODELS_BLT_MODEL_H

#include "blt/core/allocator.h"
#include "blt/core/tensor.h"
#include "blt/models/local_encoder.h"
#include "blt/models/global_transformer.h"
#include "blt/models/local_decoder.h"
#include "blt/models/patcher.h"

typedef struct {
    blt_local_encoder_config encoder_config;
    blt_global_transformer_config global_config;
    blt_local_decoder_config decoder_config;
} blt_model_config;

typedef struct {
    blt_model_config config;
    blt_local_encoder* encoder;
    blt_global_transformer* global;
    blt_local_decoder* decoder;
} blt_model;

typedef struct {
    blt_local_encoder_grad* encoder_grad;
    blt_global_transformer_grad* global_grad;
    blt_local_decoder_grad* decoder_grad;
} blt_model_grad;

blt_model* blt_model_create(blt_arena* arena, const blt_model_config* config);
blt_model_grad* blt_model_grad_create(blt_arena* arena, const blt_model* model);

void blt_model_forward(
    const blt_model* model,
    const blt_tensor* bytes_in,          // [seq_len] UINT8
    const blt_patch_info* patches,
    size_t num_patches,
    const size_t* doc_boundaries,        // byte-indexed
    size_t num_docs,
    blt_tensor* logits_out,              // [seq_len, 256]
    blt_tensor* loss_out,                // scalar
    blt_arena* arena
);

void blt_model_backward(
    const blt_model* model,
    const blt_tensor* bytes_in,
    const blt_patch_info* patches,
    size_t num_patches,
    const size_t* doc_boundaries,
    size_t num_docs,
    blt_model_grad* grad,
    blt_arena* arena
);

#endif // BLT_MODELS_BLT_MODEL_H