#ifndef BLT_MODELS_CHECKPOINT_H
#define BLT_MODELS_CHECKPOINT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "blt/models/model.h"
#include "blt/models/entropy_lm.h"

//----------------------------------------------------------------------
// Model weight serialization (checkpoints).
//
// Deterministic flat listing of every trainable tensor in a blt_model:
// encoder embedding + per-layer weights (self-attn block, then cross-
// attention block), global transformer per-layer weights, decoder
// per-layer weights + lm_head + d0_embed table. RoPE cos/sin caches are
// excluded (pure functions of config, recomputed at create time).
//
// File format (little-endian):
//   magic  "FBLT" (4 bytes)
//   u32    version (1)
//   u32    num_tensors
//   per tensor:
//     u16  name_len
//     char name[name_len]        e.g. "enc.L0.ffn_down", "dec.d0_embed"
//     u32  ndim
//     u32  dims[ndim]
//     f32  data[prod(dims)]
//
// Load validates magic, version, tensor count, every name and every
// shape against the target model (built from config), then copies
// payloads. Any mismatch is fatal (BLT_FATAL), matching repo policy.
//----------------------------------------------------------------------

// Total number of serializable tensors (config-dependent: layers).
size_t blt_model_num_tensors(const blt_model* model);

// Indexed access into the stable tensor ordering. Writes the canonical
// name pointer (static string) and the tensor handle.
void blt_model_tensor_at(const blt_model* model, size_t index,
                         const char** name, blt_tensor** tensor);

// Write all tensors to path. Fatal on IO errors.
void blt_model_save(const blt_model* model, const char* path);

// Read a checkpoint written by blt_model_save into model. Fatal on any
// format/validation mismatch.
void blt_model_load(blt_model* model, const char* path);

// Entropy-LM weight serialization (same FBLT container, tensor set:
// embedding, per-layer self-attention weights, lm_head). Load validates
// names/shapes against the target model.
void blt_entropy_lm_save(const blt_entropy_lm* lm, const char* path);
void blt_entropy_lm_load(blt_entropy_lm* lm, const char* path);

#ifdef __cplusplus
}
#endif

#endif // BLT_MODELS_CHECKPOINT_H
