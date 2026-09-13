# FBLT Test API Reference

This document describes the test harness used by the FBLT project and explains how to add new tests in a way that fits the existing structure.

## Test layout

The test system is organized into a small set of entrypoints and helper modules:

### tests/test_main.c
- Entry point for the full test binary.
- Defines the main test list and prints pass/fail results.
- Each test in the list is a function returning `int`.

### tests/test_suite.h
- Declares the top-level test runner functions.
- New test groups should be added here so they can be registered by the main test runner.

### tests/test_helpers.h and tests/test_helpers.c
- Provide reusable helpers for common test operations.
- Include assertion macros and tensor-loading helpers.

### tests/parity/
- Reference-style parity tests for the core pipeline (golden files under `data/`).
- Useful as a model for creating end-to-end tests that compare computed output to expected data.

### tests/unit/*
- Unit tests are grouped by domain:
  - `tests/unit/core/` for core tensor/backend/allocator behavior.
  - `tests/unit/backend/` for backend operator implementations.
  - `tests/unit/models/` for model-level logic such as attention, entropy, and patching.

------------------------------------------------------------------------------------------------------------
## Core test helpers

### TEST_ASSERT(cond)
- Input: a boolean condition.
- Behavior: fails the current test immediately if the condition is false.
- Use this for simple checks such as shape, dtype, or value expectations.

### TEST_ASSERT_CLOSE(actual, expected, tol)
- Input: two `blt_tensor` objects and a tolerance.
- Behavior: compares tensors element-by-element and fails if any difference exceeds `tol`.
- Use this for floating-point parity checks.

### blt_test_load_binary_tensor(path, arena, out_tensor)
- Input: a binary tensor file path, an arena, and an output tensor.
- Behavior: loads a tensor from disk into a newly allocated tensor object.
- Use this for golden-data based tests.

### blt_test_check_close(actual, expected, tol)
- Input: two tensors and a tolerance.
- Behavior: compares tensor values and reports mismatches.
- Use this when you need a direct tensor comparison helper.

### Compatibility aliases
- `load_binary_tensor` is an alias for `blt_test_load_binary_tensor`.
- `check_close` is an alias for `blt_test_check_close`.
- These aliases keep older test code readable and are still available for new tests.

------------------------------------------------------------------------------------------------------------
## Writing a new test

### 1. Choose the right location
Pick the most appropriate test directory:
- `tests/unit/core/` for tensor, allocator, or backend core utilities.
- `tests/unit/backend/` for operator implementation tests.
- `tests/unit/models/` for model behavior tests.

### 2. Create a test function
Each test should follow this pattern:
- Return `int`.
- Return `1` on success.
- Return `0` on failure.
- Use `TEST_ASSERT` or `TEST_ASSERT_CLOSE` for validation.

Example skeleton:

```c
#include "test_helpers.h"
#include "test_suite.h"

int run_my_new_tests(void) {
    blt_arena* arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    if (!arena) {
        return 0;
    }

    blt_tensor a = {0};
    blt_tensor b = {0};
    blt_tensor out = {0};

    size_t shape[2] = {2, 2};
    a = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    b = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    out = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);

    /* fill tensors here */

    /* call the implementation under test */

    /* verify results */
    TEST_ASSERT(1);

    blt_arena_destroy(arena);
    return 1;
}
```

### 3. Register the test in the suite
After creating the test function:
- Add its declaration to `tests/test_suite.h`.
- Add it to the `tests[]` array in `tests/test_main.c`.

### 4. Use the arena pattern for allocations
Most tests should:
- Create a `blt_arena` with a suitable capacity.
- Allocate tensors from that arena.
- Destroy the arena at the end of the test.

This keeps allocations simple and avoids manual memory management.

### 5. Prefer clear failure messages
When a test fails, include enough context to diagnose the problem:
- compare the expected and actual values,
- print the failing index or shape mismatch,
- confirm the test is checking the intended behavior.

------------------------------------------------------------------------------------------------------------
To write end-to-end parity tests that compare C implementation outputs against Python reference data, follow this structure:

## Python Parity Test Structure

### 1. Generating Reference (Golden) Data in Python

Reference data should be exported as binary tensors using the provided `golden` module helper functions.

1. **Script Setup**: Create a generator script (e.g., `generate_golden_*.py`) that imports `write_tensor_fp32` or `write_tensor_uint8` from `golden`.


2. **Data Generation**: Produce input data and run your Python reference model or algorithm logic to get expected output tensors.


3. **Save Binary Files**: Write inputs and expected ground-truth tensors to disk:



```python
from pathlib import Path
from golden import write_tensor_fp32, write_tensor_uint8, create_parser

def generate_data(output_dir="data"):
    out_path = Path(output_dir)
    
    # 1. Generate PyTorch or NumPy tensors
    inputs = torch.randn(2, 4, dtype=torch.float32)
    expected = inputs * 2.0  # Reference logic
    
    # 2. Write to binary format with shape headers
    write_tensor_fp32(out_path / "parity_input.bin", inputs)
    write_tensor_fp32(out_path / "parity_expected.bin", expected)

if __name__ == "__main__":
    parser = create_parser("Generate parity golden files")
    args = parser.parse_args()
    generate_data(args.output_dir)

```

### 2. Loading and Verifying in C

C parity tests load the binary golden data into `blt_tensor` objects, execute the operator under test, and check the computed outputs against the reference.

1. **Load Data**: Use `blt_test_load_binary_tensor` (or `load_binary_tensor`) to read the generated binary files from `data/` into arena-allocated tensors.


2. **Execute Operation**: Run the C implementation function passing the loaded input tensor(s).


3. **Compare Results**: Compare computed outputs against the expected ground truth tensor using `TEST_ASSERT_CLOSE` or `blt_test_check_close` with a specified tolerance.



```c
#include "test_helpers.h"
#include "test_suite.h"

int run_parity_test_example(void) {
    // 1. Allocate arena
    blt_arena* arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    if (!arena) return 0;

    // 2. Load golden tensors generated by Python
    blt_tensor input = {0};
    blt_tensor expected = {0};
    TEST_ASSERT(blt_test_load_binary_tensor("data/parity_input.bin", arena, &input));
    TEST_ASSERT(blt_test_load_binary_tensor("data/parity_expected.bin", arena, &expected));

    // 3. Create output tensor matching expected shape
    blt_tensor actual = blt_tensor_create(arena, expected.shape, expected.ndim, expected.dtype);

    /* 4. Call C implementation under test
       e.g., my_op_forward(&input, &actual);
    */

    // 5. Assert parity within floating-point tolerance
    TEST_ASSERT_CLOSE(actual, expected, 1e-4f);

    // 6. Cleanup
    blt_arena_destroy(arena);
    return 1;
}

```

### 3. Registering the Parity Test

After implementing the parity test:

1. Declare the `run_*` function in `tests/test_suite.h`.

2. Add the test runner to the execution list array in `tests/test_main.c`.

------------------------------------------------------------------------------------------------------------
## Common test patterns

### Simple value checks
Use `TEST_ASSERT` for logic checks such as:
- computed shape is correct,
- a scalar result matches an expected value,
- a condition such as `numel == 24` is satisfied.

### Tensor parity tests
Use `TEST_ASSERT_CLOSE` or `check_close(...)` when comparing computed results with a reference tensor.

### Golden-file tests
For larger reference cases:
- store input/output tensors under `data/`,
- load them with `blt_test_load_binary_tensor(...)`,
- run the implementation,
- compare against the expected tensor.

------------------------------------------------------------------------------------------------------------
## Recommended conventions

- Keep each test focused on one behavior.
- Use descriptive function names such as `run_tensor_core_tests` or `run_attention_model_tests`.
- Keep helper code local unless it is broadly reusable.
- Prefer explicit setup and teardown so failures are easy to understand.
- Add a small, readable comment block when a test uses non-obvious tensor layout or expected values.

------------------------------------------------------------------------------------------------------------
## Quick checklist for a new test

- [ ] Pick the right test directory.
- [ ] Create a `run_*` function returning `int`.
- [ ] Use `blt_arena` and `blt_tensor_create` for storage.
- [ ] Validate results with `TEST_ASSERT` or `TEST_ASSERT_CLOSE`.
- [ ] Add the function declaration to `tests/test_suite.h`.
- [ ] Register the test in `tests/test_main.c`.
