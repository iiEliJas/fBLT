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

### tests/test_phase1.c
- Contains the reference-style parity test for the Phase 1 pipeline.
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
