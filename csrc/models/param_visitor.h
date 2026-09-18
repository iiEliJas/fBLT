#ifndef BLT_MODELS_PARAM_VISITOR_H
#define BLT_MODELS_PARAM_VISITOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "models/model.h"

// Component groups for per-group diagnostics (log_component_norms, etc.).
enum {
    BLT_GROUP_EMBED = 0,
    BLT_GROUP_ENC_SELF = 1,
    BLT_GROUP_ENC_CROSS = 2,
    BLT_GROUP_GLOB = 3,
    BLT_GROUP_DEC_SELF = 4,
    BLT_GROUP_DEC_CROSS = 5,
    BLT_GROUP_HEAD = 6,
};

// Metadata passed to the visitor callback for every parameter tensor.
typedef struct {
    float *ptr;          // weight or gradient tensor pointer
    int idx;             // global parameter index (stable across calls)
    int is_grad;         // 1 = gradient tensor, 0 = weight tensor
    int is_norm;         // 1 = RMSNorm weight (norm1/norm2/cross_norm)
    int is_enc;          // belongs to encoder
    int is_glob;         // belongs to global transformer
    int is_dec;          // belongs to decoder
    int layer_idx;       // layer index within submodel
    int param_in_layer;  // index within layer (0-6 self, 0-4 cross)
    int component_group; // BLT_GROUP_*
    size_t numel;        // number of elements in tensor
    int backend;         // BLT_BACKEND_CPU or BLT_BACKEND_CUDA
} blt_param_info;

// Callback type for blt_model_visit_params().
typedef void (*blt_param_visitor_fn)(float *tensor, const blt_param_info *info, void *ctx);

// Callback type for blt_model_visit_param_pairs(): receives both weight and grad pointers.
typedef void (*blt_param_pair_visitor_fn)(float *weight, float *grad, const blt_param_info *info, void *ctx);

// Visit every parameter (or gradient) tensor in the model.
// Iteration order matches checkpoint.c blt_model_tensor_at():
//   enc.embed, enc.ngram, enc.L<i>.self(7), enc.L<i>.cross(5),
//   glob.L<i>.self(7), dec.L<i>.self(7), dec.L<i>.cross(5),
//   dec.lm_head, dec.d0_embed
//
// visit_grads: 1 = iterate gradient tensors (read from grad), 0 = weight tensors (read from model).
// When visit_grads == 0, grad may be NULL.
void blt_model_visit_params(const blt_model *model, blt_model_grad *grad, blt_param_visitor_fn fn, void *ctx,
                            int visit_grads);

// Visit each parameter ONCE, providing both weight and grad pointer.
// Same iteration order as blt_model_visit_params. info->numel and
// info->backend come from the weight tensor.
void blt_model_visit_param_pairs(const blt_model *model, blt_model_grad *grad, blt_param_pair_visitor_fn fn, void *ctx);

#ifdef __cplusplus
}
#endif
#endif // BLT_MODELS_PARAM_VISITOR_H
