#ifndef BLT_CORE_TENSOR_H
#define BLT_CORE_TENSOR_H

#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include "blt/core/dtype.h"

#define BLT_MAX_NDIM 4

typedef enum {
    BLT_BACKEND_CPU = 0,
    BLT_BACKEND_CUDA = 1
} blt_backend;

typedef struct {
    void* data;
    size_t shape[BLT_MAX_NDIM];
    size_t strides[BLT_MAX_NDIM];
    size_t ndim;
    size_t numel;
    blt_dtype dtype;
    blt_backend backend;
    bool is_view;
} blt_tensor;

size_t blt_tensor_compute_numel(const size_t* shape, size_t ndim);
void blt_tensor_compute_row_major_strides(const size_t* shape, size_t ndim, size_t* strides_out);
size_t blt_tensor_bytes(const blt_tensor* t);
void view_1d(blt_tensor* view, void* data, size_t len, blt_dtype dtype, blt_backend backend);
// Creates a 1D view of `len` elements starting `offset_elems` elements into
// src's data (dtype/backend inherited from src).
void view_1d_offset(blt_tensor* view, const blt_tensor* src, size_t offset_elems, size_t len);
void blt_tensor_view_2d(blt_tensor* t, void* data, size_t rows, size_t cols, blt_backend backend);
void blt_tensor_view_3d(blt_tensor* t, void* data, size_t d0, size_t d1, size_t d2, blt_backend backend);

void zero_tensor(blt_tensor* t);

#endif // BLT_CORE_TENSOR_H
