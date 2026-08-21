#ifndef BLT_TEST_SUITE_H
#define BLT_TEST_SUITE_H

int run_tensor_core_tests(void);
int run_allocator_core_tests(void);
int run_backend_core_tests(void);

int run_elementwise_backend_tests(void);
int run_gelu_backend_tests(void);
int run_matmul_backend_tests(void);
int run_matmul_backward_backend_tests(void);
int run_softmax_backend_tests(void);
int run_cross_entropy_tests(void);
int run_rope_backend_test(void);
int run_optim_backend_tests(void);
int run_mask_builder_backend_tests(void);

int run_entropy_model_tests(void);
int run_byte_embedding_model_tests(void);
int run_patcher_model_tests(void);
int run_attention_model_tests(void);
int run_attention_masked_model_tests(void);
int run_attention_backward_model_tests(void);
int run_cross_attention_model_tests(void);
int run_cross_attention_masked_model_tests(void);
int run_cross_attention_backward_model_tests(void);
int run_elm_model_tests(void);
int run_hash_ngram_model_tests(void);
int run_lencoder_mask_model_test(void);
int run_local_encoder_smoke_test(void);
int run_local_encoder_k_split(void);
int run_global_transformer_causal_mask(void);
int run_global_transformer_doc_boundary(void);
int run_local_decoder_cross_mask(void);
int run_local_decoder_k_split(void);

int run_matmul_parity_test(void);
int run_softmax_parity_test(void);
int run_transformer_parity_tests(void);
int run_patcher_parity_tests(void);
int run_elm_overfit_tests(void);
int run_lencoder_overfit_test(void);
int run_local_encoder_parity(void);
int run_global_transformer_parity(void);
int run_global_transformer_overfit(void);
int run_local_decoder_parity(void);
int run_local_decoder_overfit(void);
#endif // BLT_TEST_SUITE_H