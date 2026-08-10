#include "blt/core/tensor.h"

size_t blt_tensor_compute_numel(const size_t* shape, size_t ndim) {
    if (ndim == 0) {
        return 0;
    }

    size_t numel = 1;
    for (size_t i = 0; i < ndim; ++i) {
        numel *= shape[i];
    }
    return numel;
}

void blt_tensor_compute_row_major_strides(const size_t* shape, size_t ndim, size_t* strides_out) {
    if (ndim == 0) {
        return;
    }

    size_t stride = 1;
    for (size_t i = ndim; i-- > 0;) {
        strides_out[i] = stride;
        stride *= shape[i];
    }
}

size_t blt_tensor_bytes(const blt_tensor* t) {
    return t->numel * blt_dtype_sizeof(t->dtype);
}



void view_1d(blt_tensor* view, void* data, size_t len, blt_dtype dtype, blt_backend backend) {
    view->data = data;
    for (size_t d = 0; d < BLT_MAX_NDIM; ++d) {
        view->shape[d] = 0;
        view->strides[d] = 0;
    }
    view->ndim = 1;
    view->shape[0] = len;
    view->strides[0] = 1;
    view->numel = len;
    view->dtype = dtype;
    view->backend = backend;
    view->is_view = true;
}

void blt_tensor_view_2d(blt_tensor* t, void* data, size_t rows, size_t cols, blt_backend backend) {
    t->data = data;
    t->shape[0] = rows;
    t->shape[1] = cols;
    t->shape[2] = 0;
    t->shape[3] = 0;
    t->ndim = 2;
    t->numel = blt_tensor_compute_numel(t->shape, 2);
    t->dtype = BLT_DTYPE_FP32;
    t->backend = backend;
    t->is_view = true;
    blt_tensor_compute_row_major_strides(t->shape, 2, t->strides);
}
 
void blt_tensor_view_3d(blt_tensor* t, void* data, size_t d0, size_t d1, size_t d2, blt_backend backend) {
    t->data = data;
    t->shape[0] = d0;
    t->shape[1] = d1;
    t->shape[2] = d2;
    t->shape[3] = 0;
    t->ndim = 3;
    t->numel = blt_tensor_compute_numel(t->shape, 3);
    t->dtype = BLT_DTYPE_FP32;
    t->backend = backend;
    t->is_view = true;
    blt_tensor_compute_row_major_strides(t->shape, 3, t->strides);
}


void zero_tensor(blt_tensor* t) {
    memset(t->data, 0, blt_tensor_bytes(t));
}
