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


blt_tensor blt_tensor_view_2d(void* data, size_t rows, size_t cols, blt_dtype dtype){
    blt_tensor tensor = {0};
    tensor.data = data;
    tensor.shape[0] = rows;
    tensor.shape[1] = cols;
    tensor.strides[0] = cols;
    tensor.strides[1] = 1;
    tensor.ndim = 2;
    tensor.numel = rows * cols;
    tensor.dtype = dtype;
    tensor.backend = BLT_BACKEND_CPU;
    tensor.is_view = false;
    return tensor;
}
