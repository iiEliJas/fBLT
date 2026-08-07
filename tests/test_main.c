#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "blt/core/allocator.h"
#include "blt/core/backend.h"
#include "blt/ops/matmul.h"
#include "blt/ops/softmax.h"
#include "blt/models/attention.h"
#include "blt/models/transformer.h"

#include "test_helpers.h"
#include "test_suite.h"



typedef struct {
    const char* name;
    int (*fn)(void);
} test_case;



int main(void) {
    const test_case tests[] = {
        {"matmul parity", run_matmul_test},
        {"softmax parity", run_softmax_test},
        {"tensor core helpers", run_tensor_core_tests},
        {"allocator core helpers", run_allocator_core_tests},
        {"backend dispatch", run_backend_core_tests},
        {"elementwise backend", run_elementwise_backend_tests},
        {"matmul backend", run_matmul_backend_tests},
        {"softmax backend", run_softmax_backend_tests},
        {"entropy model", run_entropy_model_tests},
        {"patcher model", run_patcher_model_tests},
        {"phase1 parity", run_phase1_tests},
        {"attention model", run_attention_model_tests},
        {"phase2 parity", run_phase2_tests},
    };

    int passed = 0;
    const size_t test_count = sizeof(tests) / sizeof(tests[0]);

    printf("\n------------------------------------------\n");
    printf("            Running FBLT tests...\n");
    printf("------------------------------------------\n");

    for (size_t i = 0; i < test_count; ++i) {
        if (tests[i].fn()) {
            printf("[PASS] %s\n", tests[i].name);
            passed++;
        } else {
            printf("[FAIL] %s\n", tests[i].name);
        }
    }

    printf("------------------------------------------\n");
    printf("Summary: %d/%zu tests passed\n", passed, test_count);
    printf("------------------------------------------\n");
    
    return passed == (int)test_count ? 0 : 1;
}
