Here is the detailed architectural guide and requirements for building Phase 0 of your Byte Latent Transformer (BLT) engine. As requested, I have provided the exact header files and struct definitions while outlining the specific logical requirements for the implementation files rather than giving you the exact code.

---

## 1. Phase 0 Overview & Architecture

Every bug in deep learning implementations becomes exponentially harder to trace if you cannot trust your low-level memory allocation or matrix operations. Passing Phase 0 guarantees that any downstream issues in attention, entropy calculation, or dynamic patching are strictly algorithmic bugs rather than memory corruption or stride miscalculations.

### High-Level Architecture

In this architecture, your model code will **never** directly allocate raw heap memory or call raw CUDA API functions. Instead, all operations execute through a unified dispatch header operating on a unified `blt_tensor` C struct.

### Files to Create

* `include/blt/core/dtype.h`
* `include/blt/core/tensor.h`
* `include/blt/core/allocator.h`
* `include/blt/core/backend.h`
* `include/blt/ops/matmul.h`
* `include/blt/ops/elementwise.h`
* `include/blt/ops/softmax.h`
* `src/core/tensor.c`
* `src/core/allocator.c`
* `src/core/backend.c`
* `src/backend_cpu/matmul_cpu.c`
* `src/backend_cpu/elementwise_cpu.c`
* `src/backend_cpu/softmax_cpu.c`
* `tests/test_main.c`
* `tools/dump_reference_io.py`

---

## 2. Step 1: Data Types & Core Tensor Definition

### File 1: `include/blt/core/dtype.h`

Define the supported data types and provide an inline helper to query byte sizes.

```c
#ifndef BLT_CORE_DTYPE_H
#define BLT_CORE_DTYPE_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    BLT_DTYPE_FP32 = 0,
    BLT_DTYPE_BF16 = 1,
    BLT_DTYPE_INT32 = 2,
    BLT_DTYPE_UINT8 = 3
} blt_dtype;

static inline size_t blt_dtype_sizeof(blt_dtype dtype) {
    switch (dtype) {
        case BLT_DTYPE_FP32:  return 4;
        case BLT_DTYPE_BF16:  return 2;
        case BLT_DTYPE_INT32: return 4;
        case BLT_DTYPE_UINT8: return 1;
        default: return 0;
    }
}

#endif // BLT_CORE_DTYPE_H

```

---

### File 2: `include/blt/core/tensor.h`

The `blt_tensor` is a flat C struct carrying metadata and a raw data pointer.

```c
#ifndef BLT_CORE_TENSOR_H
#define BLT_CORE_TENSOR_H

#include <stddef.h>
#include <stdbool.h>
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

#endif // BLT_CORE_TENSOR_H

```

---

### File 3: `src/core/tensor.c` (Implementation Requirements)

* **`blt_tensor_compute_numel`**: Calculate and return the total number of elements by multiplying all dimension sizes together. Return 0 if `ndim` is 0.
* **`blt_tensor_compute_row_major_strides`**: Populate the `strides_out` array with C row-major strides. The last dimension's stride should be 1, and each preceding stride is the product of the next dimension's stride and shape.
* **`blt_tensor_bytes`**: Calculate the total memory footprint by multiplying the tensor's `numel` by the byte size of its `dtype`.

---

## 3. Step 2: High-Performance Arena/Bump Allocator

BLT uses a bump/arena allocator to prevent lock contention, memory fragmentation, and allocation latency.

### File 1: `include/blt/core/allocator.h`

```c
#ifndef BLT_CORE_ALLOCATOR_H
#define BLT_CORE_ALLOCATOR_H

#include <stddef.h>
#include "blt/core/tensor.h"

typedef struct {
    void* buffer;        
    size_t capacity;     
    size_t offset;       
    blt_backend backend; 
} blt_arena;

blt_arena* blt_arena_create(size_t capacity_bytes, blt_backend backend);
void blt_arena_destroy(blt_arena* arena);
void blt_arena_reset(blt_arena* arena);
void* blt_arena_alloc(blt_arena* arena, size_t bytes, size_t alignment);
blt_tensor blt_tensor_create(blt_arena* arena, const size_t* shape, size_t ndim, blt_dtype dtype);

#endif // BLT_CORE_ALLOCATOR_H

```

---

### File 2: `src/core/allocator.c` (Implementation Requirements)

* **`blt_arena_create`**: Allocate the `blt_arena` struct. If the backend is CPU, allocate the main buffer using an aligned allocation method (e.g., `posix_memalign` or `_aligned_malloc`) with a 64-byte alignment. If CUDA is requested, error out (to be added in Phase 7).
* **`blt_arena_destroy`**: Free the aligned memory buffer and the struct itself.
* **`blt_arena_reset`**: Reset the arena's offset to 0 ($O(1)$ time).


* **`blt_arena_alloc`**: Align the current offset to the requested alignment. Check if the new offset plus requested bytes exceeds `capacity`. If so, throw a fatal error. Otherwise, return the pointer to the aligned offset and update the arena's offset.
* **`blt_tensor_create`**: Initialize a `blt_tensor` struct. Copy the shape, compute `numel` and `strides`, and set metadata. Calculate required bytes, call `blt_arena_alloc`, zero out the allocated memory, and assign it to the tensor's `data` pointer.

---

## 4. Step 3: Backend Dispatch Mechanism

### File 1: `include/blt/core/backend.h`

```c
#ifndef BLT_CORE_BACKEND_H
#define BLT_CORE_BACKEND_H

#include <stdio.h>
#include <stdlib.h>

#define BLT_FATAL(msg) do { \
    fprintf(stderr, "BLT Fatal Error [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
    exit(EXIT_FAILURE); \
} while(0)

#endif // BLT_CORE_BACKEND_H

```

---

### Public Operation Headers

**`include/blt/ops/elementwise.h`**

```c
#ifndef BLT_OPS_ELEMENTWISE_H
#define BLT_OPS_ELEMENTWISE_H
#include "blt/core/tensor.h"
void blt_add(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
void blt_mul(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
#endif

```

**`include/blt/ops/matmul.h`**

```c
#ifndef BLT_OPS_MATMUL_H
#define BLT_OPS_MATMUL_H
#include "blt/core/tensor.h"
void blt_matmul(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
#endif

```

**`include/blt/ops/softmax.h`**

```c
#ifndef BLT_OPS_SOFTMAX_H
#define BLT_OPS_SOFTMAX_H
#include "blt/core/tensor.h"
void blt_softmax(const blt_tensor* in, blt_tensor* out);
#endif

```

---

### Dispatch File: `src/core/backend.c` (Implementation Requirements)

* **Forward Declarations**: Declare CPU implementations (e.g., `blt_add_cpu`). Use preprocessor directives (`#ifdef BLT_WITH_CUDA`) to optionally declare CUDA implementations.


* **Dispatch Logic**: For each public function (`blt_add`, `blt_matmul`, etc.), check the input tensor's `backend` tag.


* If `BLT_BACKEND_CUDA`, verify `BLT_WITH_CUDA` is defined and call the CUDA variant; otherwise, trigger `BLT_FATAL`.
* If `BLT_BACKEND_CPU`, route to the CPU variant.

---

## 5. Step 4: CPU Fundamental Operations

Write the reference CPU code inside `src/backend_cpu/`.

### File 1: `src/backend_cpu/elementwise_cpu.c` (Implementation Requirements)

* **`blt_add_cpu` & `blt_mul_cpu**`: Verify all tensors have the exact same `numel`. Verify all tensors are `BLT_DTYPE_FP32`. Iterate from 0 to `numel`, performing the respective mathematical operation element-by-element and storing the result in the output tensor.

### File 2: `src/backend_cpu/matmul_cpu.c` (Implementation Requirements)

* **`blt_matmul_cpu`**: Validate that tensors `a`, `b`, and `out` are all exactly 2D. Validate inner matrix dimensions match ($K$ of matrix A matches $K$ of matrix B). Implement a standard triple-nested loop to compute the matrix multiplication in single precision.



### File 3: `src/backend_cpu/softmax_cpu.c` (Implementation Requirements)

* **`blt_softmax_cpu`**: Calculate softmax across the last dimension of the input tensor. For numerical stability, implement this safely: find the maximum value in the row, subtract it from each element before exponentiating ($x_i \leftarrow x_i - \max(x)$), sum the exponentiated values, and divide each element by that sum.



---

## 6. Step 5: Python Reference Data Exporter

### File: `tools/dump_reference_io.py` (Implementation Requirements)

Create a Python script to generate binary verification packages using PyTorch.

* **Binary Format**: Write a helper function that exports a tensor to a file in this exact structure:
1. `uint32_t` representing `ndim`.
2. `uint32_t` array representing the shape dimensions.
3. Raw `float32` byte array of the tensor data.


* **Generators**: Implement functions to generate random float32 tensors, compute PyTorch matrix multiplication and softmax, and save the inputs and outputs to files like `tests/golden_matmul.bin` and `tests/golden_softmax.bin`.

---

## 7. Step 6: Custom Testing Harness

### File: `tests/test_main.c` (Implementation Requirements)

Write a minimal C test runner.

* **Loader Function**: Write a helper to read the custom binary format, create an appropriately shaped tensor via the arena, and read the float data into the tensor's buffer.
* **Parity Checker**: Write a function that iterates through a computed tensor and an expected tensor, ensuring the absolute difference between each element is below a provided tolerance.
* **Test Cases**: Implement individual test functions that open the golden binary files, run your custom `blt_matmul` and `blt_softmax` functions, and assert that the results match expected values. Print timing and pass/fail status.