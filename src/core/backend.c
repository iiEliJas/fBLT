#include "blt/core/backend.h"
#include "blt/ops/elementwise.h"
#include "blt/ops/matmul.h"
#include "blt/ops/softmax.h"
#include "blt/core/tensor.h"

void blt_add_cpu(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
void blt_mul_cpu(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
void blt_matmul_cpu(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
void blt_softmax_cpu(const blt_tensor* in, blt_tensor* out);

void blt_add(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    if (a->backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        /* CUDA hook placeholder */
#else
        BLT_FATAL("CUDA backend requested but BLT_WITH_CUDA is not enabled");
#endif
    }
    blt_add_cpu(a, b, out);
}

void blt_mul(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    if (a->backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        /* CUDA hook placeholder */
#else
        BLT_FATAL("CUDA backend requested but BLT_WITH_CUDA is not enabled");
#endif
    }
    blt_mul_cpu(a, b, out);
}

void blt_matmul(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    if (a->backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        /* CUDA hook placeholder */
#else
        BLT_FATAL("CUDA backend requested but BLT_WITH_CUDA is not enabled");
#endif
    }
    blt_matmul_cpu(a, b, out);
}

void blt_softmax(const blt_tensor* in, blt_tensor* out) {
    if (in->backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        /* CUDA hook placeholder */
#else
        BLT_FATAL("CUDA backend requested but BLT_WITH_CUDA is not enabled");
#endif
    }
    blt_softmax_cpu(in, out);
}
