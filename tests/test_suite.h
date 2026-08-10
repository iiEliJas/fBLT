#ifndef BLT_TEST_SUITE_H
#define BLT_TEST_SUITE_H

int run_tensor_core_tests(void);
int run_allocator_core_tests(void);
int run_backend_core_tests(void);

int run_elementwise_backend_tests(void);
int run_gelu_backend_tests(void);
int run_matmul_backend_tests(void);
int run_matmul_test(void);
int run_matmul_backward_backend_tests(void);
int run_softmax_backend_tests(void);
int run_softmax_test(void);
int run_cross_entropy_tests(void);
int run_rope_backend_test(void);
int run_optim_backend_tests(void);

int run_entropy_model_tests(void);
int run_byte_embedding_model_tests(void);
int run_patcher_model_tests(void);
int run_attention_model_tests(void);
int run_attention_backward_model_tests(void);
int run_entropy_lm_tests(void);

int run_phase2_attention_parity_test(void);
int run_phase2_attention_backward_parity_test(void);
int run_phase2_transformer_block_parity_test(void);

int run_phase2_tests(void);
int run_phase1_tests(void);
int run_overfit_entropy_lm_tests(void);
#endif // BLT_TEST_SUITE_H