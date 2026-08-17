#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "blt/core/allocator.h"
#include "blt/core/backend.h"

#include "test_helpers.h"
#include "test_suite.h"



typedef struct {
    const char* name;
    int (*fn)(void);
} test_case;



int main(void) {
    const test_case core_tests[] = {
        {"tensor core helpers", run_tensor_core_tests},
        {"allocator core helpers", run_allocator_core_tests},
        {"backend dispatch", run_backend_core_tests},

        {"elementwise backend", run_elementwise_backend_tests},
        {"gelu backend", run_gelu_backend_tests},
        {"matmul backend", run_matmul_backend_tests},
        {"matmul backward backend", run_matmul_backward_backend_tests},
        {"softmax backend", run_softmax_backend_tests},
        {"cross entropy backend", run_cross_entropy_tests},
        {"rope backend", run_rope_backend_test},
        {"optimizer backend", run_optim_backend_tests},
        {"mask builder backend", run_mask_builder_backend_tests},  
    };

    const test_case model_tests[] = {
        {"entropy model", run_entropy_model_tests},
        {"patcher model", run_patcher_model_tests},
        {"byte embedding model", run_byte_embedding_model_tests},
        {"attention model", run_attention_model_tests},
        {"attention masked model", run_attention_masked_model_tests},
        {"attention backward model", run_attention_backward_model_tests},
        {"cross attention model", run_cross_attention_model_tests},
        {"cross attention masked model", run_cross_attention_masked_model_tests},
        {"cross attention backward model", run_cross_attention_backward_model_tests},
        {"entropy LM model", run_elm_model_tests},
        {"hash ngram model", run_hash_ngram_model_tests},
    };

    const test_case parity_tests[] = {
        {"matmul parity", run_matmul_parity_test},
        {"softmax parity", run_softmax_parity_test},
        {"patcher parity", run_patcher_parity_tests},
        {"transformer parity", run_transformer_parity_tests},
        {"entropy LM overfit", run_elm_overfit_tests},
    };

    int passed = 0;

    printf("\n------------------------------------------\n");
    printf("            Running FBLT tests...\n");
    printf("------------------------------------------\n");

    printf("\n------------------------\n");
    printf("Running core tests...\n\n");

    for (size_t i = 0; i < sizeof(core_tests) / sizeof(core_tests[0]); ++i) {
        if (core_tests[i].fn()) {
            printf("[PASS] %s\n", core_tests[i].name);
            passed++;
        } else {
            printf("[FAIL] %s\n", core_tests[i].name);
        }
    }

    printf("\n------------------------\n");
    printf("Running model tests...\n\n");

    for (size_t i = 0; i < sizeof(model_tests) / sizeof(model_tests[0]); ++i) {
        if (model_tests[i].fn()) {
            printf("[PASS] %s\n", model_tests[i].name);
            passed++;
        } else {
            printf("[FAIL] %s\n", model_tests[i].name);
        }
    }

    printf("\n------------------------\n");
    printf("Running parity tests...\n\n");

    for (size_t i = 0; i < sizeof(parity_tests) / sizeof(parity_tests[0]); ++i) {
        if (parity_tests[i].fn()) {
            printf("[PASS] %s\n", parity_tests[i].name);
            passed++;
        } else {
            printf("[FAIL] %s\n", parity_tests[i].name);
        }
    }

    size_t test_count = sizeof(core_tests) / sizeof(core_tests[0]) + sizeof(model_tests) / sizeof(model_tests[0]) + sizeof(parity_tests) / sizeof(parity_tests[0]);
    printf("\n------------------------------------------\n");
    printf("Summary: %d/%zu tests passed\n", passed, test_count);
    printf("------------------------------------------\n");
    
    return passed == (int)test_count ? 0 : 1;
}
