#ifndef BLT_TEST_SUITE_H
#define BLT_TEST_SUITE_H

int run_tensor_core_tests(void);
int run_allocator_core_tests(void);
int run_backend_core_tests(void);
int run_elementwise_backend_tests(void);
int run_matmul_backend_tests(void);
int run_softmax_backend_tests(void);
int run_entropy_model_tests(void);
int run_patcher_model_tests(void);
int run_phase1_tests(void);
int run_attention_model_tests(void);

#endif // BLT_TEST_SUITE_H