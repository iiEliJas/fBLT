#ifndef BLT_TRAIN_DIAG_H
#define BLT_TRAIN_DIAG_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "core/tensor.h"
#include "models/block_diffusion.h"
#include "models/model.h"
#include "models/param_visitor.h"
#include "models/patcher.h"

// Gradient clipping: clip all gradients to max_norm and return the pre-clip norm.
float clip_all(blt_model *m, blt_model_grad *g, float max_norm);

// Log per-component L2 gradient norms to the component-norm-log file.
void log_component_norms(FILE *fp, size_t step, blt_model *m, blt_model_grad *g);

// Log forward activation stats (before backward). NULL tensors are skipped.
void log_forward_activation_dump(FILE *fp, size_t step, const blt_tensor *P, const blt_tensor *h, const blt_tensor *O,
                                 const blt_tensor *logits);

// Log gradient activation dump for numerical anomaly detection.
void log_gradient_activation_dump(FILE *fp, size_t step, blt_model *m, blt_model_grad *g);

// Log per-step batch properties: window identity, patch stats, diffusion
// corruption stats, and the effective loss weight.
void log_batch_properties(FILE *fp, size_t step, size_t window_offset, const uint8_t *text, size_t window_size,
                          const blt_patch_info *patches, size_t num_patches, const blt_block_batch *batch);

// Visitor callback: zero out RMSNorm weights. Stays as declaration for use in train_blt_d.c.
void zero_norm_fn(float *p, const blt_param_info *info, void *ctx);

#ifdef __cplusplus
}
#endif
#endif // BLT_TRAIN_DIAG_H
