#include "blt/core/backend.h"
#include "blt/ops/elementwise.h"
#include "blt/ops/matmul.h"
#include "blt/ops/softmax.h"
#include "blt/ops/vecmath.h"
#include "blt/ops/gelu.h"
#include "blt/core/tensor.h"

void blt_add_cpu(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
void blt_mul_cpu(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
void blt_matmul_cpu(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
void blt_softmax_cpu(const blt_tensor* in, blt_tensor* out);
void blt_gelu_forward_cpu(const blt_tensor* x, blt_tensor* out);
void blt_gelu_backward_cpu(const blt_tensor* grad_out, const blt_tensor* x, blt_tensor* grad_x);


// ------------------------------------------------------------
// Placeholder helper for unwritten CUDA ops

#define BLT_CUDA_NOT_IMPLEMENTED(op_name) \
    BLT_FATAL(op_name " is not yet implemented for CUDA")

/* Dispatch Macros */
#ifdef BLT_WITH_CUDA
#define BLT_DISPATCH(tensor, cpu_call, cuda_call)               \
    do {                                                         \
        if ((tensor)->backend == BLT_BACKEND_CUDA) {             \
            cuda_call;                                           \
            return;                                              \
        }                                                        \
        cpu_call;                                                \
    } while (0)

#define BLT_DISPATCH_RET(tensor, cpu_call, cuda_call)           \
    do {                                                         \
        if ((tensor)->backend == BLT_BACKEND_CUDA) {             \
            return cuda_call;                                    \
        }                                                        \
        return cpu_call;                                         \
    } while (0)
#else
#define BLT_DISPATCH(tensor, cpu_call, cuda_call)               \
    do {                                                         \
        if ((tensor)->backend == BLT_BACKEND_CUDA) {             \
            BLT_FATAL("CUDA backend requested but BLT_WITH_CUDA is not enabled"); \
        }                                                        \
        cpu_call;                                                \
    } while (0)

#define BLT_DISPATCH_RET(tensor, cpu_call, cuda_call)           \
    do {                                                         \
        if ((tensor)->backend == BLT_BACKEND_CUDA) {             \
            BLT_FATAL("CUDA backend requested but BLT_WITH_CUDA is not enabled"); \
        }                                                        \
        return cpu_call;                                         \
    } while (0)
#endif



// ------------------------------------------------------------
// BLT Backend Operations

void blt_add(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    BLT_DISPATCH(a, blt_add_cpu(a, b, out), BLT_CUDA_NOT_IMPLEMENTED("blt_add"));
}

void blt_mul(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    BLT_DISPATCH(a, blt_mul_cpu(a, b, out), BLT_CUDA_NOT_IMPLEMENTED("blt_mul"));
}

void blt_matmul(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    BLT_DISPATCH(a, blt_matmul_cpu(a, b, out), BLT_CUDA_NOT_IMPLEMENTED("blt_matmul"));
}

void blt_softmax(const blt_tensor* in, blt_tensor* out) {
    BLT_DISPATCH(in, blt_softmax_cpu(in, out), BLT_CUDA_NOT_IMPLEMENTED("blt_softmax"));
}

void blt_gelu_forward(const blt_tensor* x, blt_tensor* out) {
    BLT_DISPATCH(x, blt_gelu_forward_cpu(x, out), BLT_CUDA_NOT_IMPLEMENTED("blt_gelu_forward"));
}

void blt_gelu_backward(const blt_tensor* grad_out, const blt_tensor* x, blt_tensor* grad_x) {
    BLT_DISPATCH(x, blt_gelu_backward_cpu(grad_out, x, grad_x), BLT_CUDA_NOT_IMPLEMENTED("blt_gelu_backward"));
}
