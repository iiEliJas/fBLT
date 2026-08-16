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
    const test_case tests[] = {
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
    
        {"entropy model", run_entropy_model_tests},
        {"patcher model", run_patcher_model_tests},
        {"byte embedding model", run_byte_embedding_model_tests},
        {"attention model", run_attention_model_tests},
        {"attention masked model", run_attention_masked_model_tests},
        {"attention backward model", run_attention_backward_model_tests},
        {"entropy LM model", run_elm_model_tests},
        {"hash ngram model", run_hash_ngram_model_tests},

        {"matmul parity", run_matmul_parity_test},
        {"softmax parity", run_softmax_parity_test},
        {"patcher parity", run_patcher_parity_tests},
        {"transformer parity", run_transformer_parity_tests},
        {"entropy LM overfit", run_elm_overfit_tests},
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
