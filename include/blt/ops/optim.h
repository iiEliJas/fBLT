#ifndef BLT_OPS_OPTIM_H
#define BLT_OPS_OPTIM_H

#include "blt/core/tensor.h"

// In-place SGD parameter update: param -= lr * grad.
void blt_sgd_step(blt_tensor* param, const blt_tensor* grad, float lr);

#endif // BLT_OPS_OPTIM_H