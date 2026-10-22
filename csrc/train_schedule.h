#ifndef BLT_TRAIN_SCHEDULE_H
#define BLT_TRAIN_SCHEDULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "train_args.h"

// Compute the learning rate for a given optimizer step.
// Pure function of args + step + end_step; unit-testable without a model.
float train_lr_at_step(const args_t *a, size_t step, size_t end_step);

#ifdef __cplusplus
}
#endif

#endif // BLT_TRAIN_SCHEDULE_H