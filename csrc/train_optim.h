#ifndef BLT_TRAIN_OPTIM_H
#define BLT_TRAIN_OPTIM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "models/model.h"
#include "models/param_visitor.h"
#include "ops/optim.h"
#include "core/allocator.h"

#ifndef BLT_MAX_PARAMS
#define BLT_MAX_PARAMS 4096
#endif

// SGD: w -= lr * g for every parameter
void sgd_all(blt_model *m, blt_model_grad *g, float lr);

// AdamW state: one exp_avg + one exp_avg_sq per parameter, flat array indexed by param idx
typedef struct {
    blt_tensor em;
    blt_tensor esq;
} adamw_pair;

typedef struct {
    adamw_pair flat[BLT_MAX_PARAMS];
    size_t n;
    size_t step;
} adamw_state;

// Create adamw_state with flat pairs for every model parameter
adamw_state *adamw_state_create(blt_arena *arena, blt_model *m);

// AdamW update step: w -= lr * bias_corrected_adamw(w, g, m, v, ...)
void adamw_all(blt_model *m, blt_model_grad *g, adamw_state *s, const blt_adamw_config *cfg);

// Compute ||update|| across all parameters (for gradient clipping diagnostics)
float adamw_update_norm(blt_model *m, blt_model_grad *g, adamw_state *s, const blt_adamw_config *cfg);

#ifdef __cplusplus
}
#endif
#endif // BLT_TRAIN_OPTIM_H
