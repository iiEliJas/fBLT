#include "blt/core/backend.h"
#include "blt/ops/elementwise.h"
#include "blt/ops/matmul.h"
#include "blt/ops/softmax.h"
#include "blt/ops/rope.h"
#include "blt/ops/layernorm.h"
#include "blt/ops/rmsnorm.h"
#include "blt/ops/vecmath.h"
#include "blt/ops/gelu.h"
#include "blt/ops/swiglu.h"
#include "blt/ops/cross_entropy.h"
#include "blt/ops/optim.h"
#include "blt/ops/mask_builder.h"
#include "blt/ops/patch_pool.h"
#include "blt/core/tensor.h"
// ------------------------------------------------------------
// CPU Implementation Declarations

// elementwise ops
void blt_add_cpu(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
void blt_mul_cpu(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
void blt_gelu_forward_cpu(const blt_tensor* x, blt_tensor* out);
void blt_gelu_backward_cpu(const blt_tensor* grad_out, const blt_tensor* x, blt_tensor* grad_x);
void blt_swiglu_forward_cpu(const blt_tensor* gate, const blt_tensor* up, blt_tensor* out);
void blt_swiglu_backward_cpu(const blt_tensor* grad_out, const blt_tensor* gate, 
                            const blt_tensor* up, blt_tensor* grad_gate, blt_tensor* grad_up);

// linear algebra ops
void blt_matmul_cpu(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
void blt_matmul_backward_cpu(const blt_tensor* a, const blt_tensor* b, const blt_tensor* grad_out,
                          blt_tensor* grad_a, blt_tensor* grad_b);

// vecmath ops (backend-keyed raw-pointer utilities)
float blt_vec_dot_cpu(blt_backend backend, const float* a, const float* b, size_t n);
void blt_softmax_masked_row_inplace_cpu(blt_backend backend, float* row, size_t row_len,
                                        size_t row_idx, bool is_causal,
                                        const float* mask_row, float scale);

// in-place scalar scale (elementwise)
void blt_scale_cpu(blt_tensor* t, float scalar);

// optimizers
void blt_sgd_step_cpu(blt_tensor* param, const blt_tensor* grad, float lr);
void blt_adamw_step_cpu(blt_tensor* param, const blt_tensor* grad,
                        blt_tensor* exp_avg, blt_tensor* exp_avg_sq,
                        const blt_adamw_config* config);

// mask builder
void blt_build_attention_mask_cpu(const blt_mask_config* config, blt_tensor* out_mask, blt_arena* arena);
void blt_build_block_diffusion_mask_cpu(const blt_block_diffusion_config* config,
                                        blt_tensor* out_mask, blt_arena* arena);

// patch pool (group-id builders are host-array utilities, not dispatched)
void blt_patch_pool_forward_cpu(const blt_tensor* byte_hidden, const blt_patch_info* patches,
                                size_t num_patches, blt_patch_pool_type pool_type, blt_tensor* out);
void blt_patch_pool_backward_cpu(const blt_tensor* grad_out, const blt_tensor* byte_hidden,
                                 const blt_patch_info* patches, size_t num_patches,
                                 blt_patch_pool_type pool_type, blt_tensor* grad_byte_hidden);

// reduction ops
void blt_softmax_cpu(const blt_tensor* in, blt_tensor* out);
void blt_softmax_backward_cpu(const blt_tensor* grad_out, const blt_tensor* softmax_out, blt_tensor* grad_in);
void blt_cross_entropy_forward_cpu(const blt_tensor* logits, const blt_tensor* targets, blt_tensor* loss_out);
void blt_cross_entropy_backward_cpu(const blt_tensor* logits, const blt_tensor* targets, blt_tensor* grad_logits);
void blt_rope_precompute_cpu(size_t max_seq_len, const blt_rope_config* config, blt_tensor* cos_out, blt_tensor* sin_out);
void blt_rope_apply_cpu(const blt_tensor* x, const blt_tensor* cos, const blt_tensor* sin, blt_tensor* out);
void blt_rope_apply_backward_cpu(const blt_tensor* grad_out, const blt_tensor* cos, const blt_tensor* sin, blt_tensor* grad_in);
void blt_layernorm_forward_cpu(const blt_tensor* x, const blt_tensor* weight, const blt_tensor* bias,blt_tensor* out, float eps);
void blt_layernorm_backward_cpu(const blt_tensor* grad_out, const blt_tensor* x,const blt_tensor* weight, blt_tensor* grad_x,
                            blt_tensor* grad_weight, blt_tensor* grad_bias, float eps);
void blt_rmsnorm_forward_cpu(const blt_tensor* x, const blt_tensor* weight, blt_tensor* out);
void blt_rmsnorm_backward_cpu(const blt_tensor* grad_out, const blt_tensor* x,
                           const blt_tensor* weight, blt_tensor* grad_x, blt_tensor* grad_weight);


// ------------------------------------------------------------
// Placeholder helper for unwritten CUDA ops

#define BLT_CUDA_NOT_IMPLEMENTED(op_name) \
    BLT_FATAL(op_name " is not yet implemented for CUDA")

/* Dispatch Macros */
#ifdef BLT_WITH_CUDA
#define BLT_DISPATCH(tensor, cpu_call, cuda_call)               \
    do {                                                         \
        if ((tensor)->backend == BLT_BACKEND_CUDA) {             \
            cuda_call;                                           \
            return;                                              \
        }                                                        \
        cpu_call;                                                \
    } while (0)

#define BLT_DISPATCH_RET(tensor, cpu_call, cuda_call)           \
    do {                                                         \
        if ((tensor)->backend == BLT_BACKEND_CUDA) {             \
            return cuda_call;                                    \
        }                                                        \
        return cpu_call;                                         \
    } while (0)

// Backend-keyed variant for raw-pointer utilities that have no tensor handle.
#define BLT_DISPATCH_BACKEND(backend, cpu_call, cuda_call)      \
    do {                                                         \
        if ((backend) == BLT_BACKEND_CUDA) {                     \
            cuda_call;                                           \
            return;                                              \
        }                                                        \
        cpu_call;                                                \
    } while (0)

#define BLT_DISPATCH_BACKEND_RET(backend, cpu_call, cuda_call)  \
    do {                                                         \
        if ((backend) == BLT_BACKEND_CUDA) {                     \
            return cuda_call;                                    \
        }                                                        \
        return cpu_call;                                         \
    } while (0)

float blt_vec_dot_cuda(blt_backend backend, const float* a, const float* b, size_t n);
void blt_softmax_masked_row_inplace_cuda(blt_backend backend, float* row, size_t row_len,
                                         size_t row_idx, int is_causal,
                                         const float* mask_row, float scale);

// elementwise ops
void blt_add_cuda(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
void blt_mul_cuda(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
void blt_scale_cuda(blt_tensor* t, float scalar);
void blt_gelu_forward_cuda(const blt_tensor* x, blt_tensor* out);
void blt_gelu_backward_cuda(const blt_tensor* grad_out, const blt_tensor* x, blt_tensor* grad_x);
void blt_swiglu_forward_cuda(const blt_tensor* gate, const blt_tensor* up, blt_tensor* out);
void blt_swiglu_backward_cuda(const blt_tensor* grad_out, const blt_tensor* gate,
                              const blt_tensor* up, blt_tensor* grad_gate, blt_tensor* grad_up);

// reduction ops
void blt_softmax_cuda(const blt_tensor* in, blt_tensor* out);
void blt_softmax_backward_cuda(const blt_tensor* grad_out, const blt_tensor* softmax_out, blt_tensor* grad_in);
void blt_cross_entropy_forward_cuda(const blt_tensor* logits, const blt_tensor* targets, blt_tensor* loss_out);
void blt_cross_entropy_backward_cuda(const blt_tensor* logits, const blt_tensor* targets, blt_tensor* grad_logits);
void blt_rope_precompute_cuda(size_t max_seq_len, const blt_rope_config* config, blt_tensor* cos_out, blt_tensor* sin_out);
void blt_rope_apply_cuda(const blt_tensor* x, const blt_tensor* cos, const blt_tensor* sin, blt_tensor* out);
void blt_rope_apply_backward_cuda(const blt_tensor* grad_out, const blt_tensor* cos, const blt_tensor* sin, blt_tensor* grad_in);
void blt_layernorm_forward_cuda(const blt_tensor* x, const blt_tensor* weight, const blt_tensor* bias, blt_tensor* out, float eps);
void blt_rmsnorm_forward_cuda(const blt_tensor* x, const blt_tensor* weight, blt_tensor* out);
void blt_rmsnorm_backward_cuda(const blt_tensor* grad_out, const blt_tensor* x,
                               const blt_tensor* weight, blt_tensor* grad_x, blt_tensor* grad_weight);

// linear algebra ops
void blt_matmul_cuda(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
void blt_matmul_backward_cuda(const blt_tensor* a, const blt_tensor* b, const blt_tensor* grad_out,
                              blt_tensor* grad_a, blt_tensor* grad_b);

// optimizers
void blt_sgd_step_cuda(blt_tensor* param, const blt_tensor* grad, float lr);
void blt_adamw_step_cuda(blt_tensor* param, const blt_tensor* grad,
                         blt_tensor* exp_avg, blt_tensor* exp_avg_sq,
                         const blt_adamw_config* config);

// mask builder
void blt_build_attention_mask_cuda(const blt_mask_config* config, blt_tensor* out_mask, blt_arena* arena);
void blt_build_block_diffusion_mask_cuda(const blt_block_diffusion_config* config,
                                         blt_tensor* out_mask, blt_arena* arena);

// patch pool
void blt_patch_pool_forward_cuda(const blt_tensor* byte_hidden, const blt_patch_info* patches,
                                 size_t num_patches, blt_patch_pool_type pool_type, blt_tensor* out);
void blt_patch_pool_backward_cuda(const blt_tensor* grad_out, const blt_tensor* byte_hidden,
                                  const blt_patch_info* patches, size_t num_patches,
                                  blt_patch_pool_type pool_type, blt_tensor* grad_byte_hidden);
#else
#define BLT_DISPATCH(tensor, cpu_call, cuda_call)               \
    do {                                                         \
        if ((tensor)->backend == BLT_BACKEND_CUDA) {             \
            BLT_FATAL("CUDA backend requested but BLT_WITH_CUDA is not enabled"); \
        }                                                        \
        cpu_call;                                                \
    } while (0)

#define BLT_DISPATCH_RET(tensor, cpu_call, cuda_call)           \
    do {                                                         \
        if ((tensor)->backend == BLT_BACKEND_CUDA) {             \
            BLT_FATAL("CUDA backend requested but BLT_WITH_CUDA is not enabled"); \
        }                                                        \
        return cpu_call;                                         \
    } while (0)

#define BLT_DISPATCH_BACKEND(backend, cpu_call, cuda_call)      \
    do {                                                         \
        if ((backend) == BLT_BACKEND_CUDA) {                     \
            BLT_FATAL("CUDA backend requested but BLT_WITH_CUDA is not enabled"); \
        }                                                        \
        cpu_call;                                                \
    } while (0)

#define BLT_DISPATCH_BACKEND_RET(backend, cpu_call, cuda_call)  \
    do {                                                         \
        if ((backend) == BLT_BACKEND_CUDA) {                     \
            BLT_FATAL("CUDA backend requested but BLT_WITH_CUDA is not enabled"); \
        }                                                        \
        return cpu_call;                                         \
    } while (0)
#endif



// ------------------------------------------------------------
// BLT Backend Operations

// ------------------
// ELEMENTWISE

void blt_add(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    BLT_DISPATCH(a, blt_add_cpu(a, b, out), blt_add_cuda(a, b, out));
}

void blt_mul(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    BLT_DISPATCH(a, blt_mul_cpu(a, b, out), blt_mul_cuda(a, b, out));
}

void blt_scale(blt_tensor* t, float scalar) {
    BLT_DISPATCH(t, blt_scale_cpu(t, scalar), blt_scale_cuda(t, scalar));
}

void blt_gelu_forward(const blt_tensor* x, blt_tensor* out) {
    BLT_DISPATCH(x, blt_gelu_forward_cpu(x, out), blt_gelu_forward_cuda(x, out));
}

void blt_gelu_backward(const blt_tensor* grad_out, const blt_tensor* x, blt_tensor* grad_x) {
    BLT_DISPATCH(x, blt_gelu_backward_cpu(grad_out, x, grad_x), blt_gelu_backward_cuda(grad_out, x, grad_x));
}

void blt_swiglu_forward(const blt_tensor* gate, const blt_tensor* up, blt_tensor* out) {
    BLT_DISPATCH(gate, blt_swiglu_forward_cpu(gate, up, out), blt_swiglu_forward_cuda(gate, up, out));
}

void blt_swiglu_backward(const blt_tensor* grad_out, const blt_tensor* gate, const blt_tensor* up, blt_tensor* grad_gate, blt_tensor* grad_up) {
    BLT_DISPATCH(gate, blt_swiglu_backward_cpu(grad_out, gate, up, grad_gate, grad_up),
                 blt_swiglu_backward_cuda(grad_out, gate, up, grad_gate, grad_up));
}



// ------------------
// REDUCTIONS

void blt_softmax(const blt_tensor* in, blt_tensor* out) {
    BLT_DISPATCH(in, blt_softmax_cpu(in, out), blt_softmax_cuda(in, out));
}

void blt_softmax_backward(const blt_tensor* grad_out, const blt_tensor* softmax_out, blt_tensor* grad_in){
    BLT_DISPATCH(grad_out, blt_softmax_backward_cpu(grad_out, softmax_out, grad_in),
                 blt_softmax_backward_cuda(grad_out, softmax_out, grad_in));
}

void blt_cross_entropy_forward(const blt_tensor* logits, const blt_tensor* targets, blt_tensor* loss_out){
    BLT_DISPATCH(logits, blt_cross_entropy_forward_cpu(logits, targets, loss_out),
                 blt_cross_entropy_forward_cuda(logits, targets, loss_out));
}

void blt_cross_entropy_backward(const blt_tensor* logits, const blt_tensor* targets, blt_tensor* grad_logits){
    BLT_DISPATCH(logits, blt_cross_entropy_backward_cpu(logits, targets, grad_logits),
                 blt_cross_entropy_backward_cuda(logits, targets, grad_logits));
}

void blt_rope_precompute(size_t max_seq_len, const blt_rope_config* config, blt_tensor* cos_out, blt_tensor* sin_out) {
    BLT_DISPATCH(cos_out, blt_rope_precompute_cpu(max_seq_len, config, cos_out, sin_out),
                 blt_rope_precompute_cuda(max_seq_len, config, cos_out, sin_out));
}

void blt_rope_apply(const blt_tensor* x, const blt_tensor* cos, const blt_tensor* sin, blt_tensor* out) {
    BLT_DISPATCH(x, blt_rope_apply_cpu(x, cos, sin, out), blt_rope_apply_cuda(x, cos, sin, out));
}

void blt_rope_apply_backward(const blt_tensor* grad_out, const blt_tensor* cos, const blt_tensor* sin, blt_tensor* grad_in){
    BLT_DISPATCH(grad_out, blt_rope_apply_backward_cpu(grad_out, cos, sin, grad_in),
                 blt_rope_apply_backward_cuda(grad_out, cos, sin, grad_in));
}

void blt_layernorm_forward(const blt_tensor* x, const blt_tensor* weight, const blt_tensor* bias,blt_tensor* out, float eps){
    BLT_DISPATCH(x, blt_layernorm_forward_cpu(x, weight, bias, out, eps),
                 blt_layernorm_forward_cuda(x, weight, bias, out, eps));
}
void blt_layernorm_backward(const blt_tensor* grad_out, const blt_tensor* x,const blt_tensor* weight, blt_tensor* grad_x, blt_tensor* grad_weight, blt_tensor* grad_bias, float eps){
    BLT_DISPATCH(x, blt_layernorm_backward_cpu(grad_out, x, weight, grad_x, grad_weight, grad_bias, eps), BLT_CUDA_NOT_IMPLEMENTED("blt_layernorm_backward"));
}

void blt_rmsnorm_forward(const blt_tensor* x, const blt_tensor* weight, blt_tensor* out){
    BLT_DISPATCH(x, blt_rmsnorm_forward_cpu(x, weight, out), blt_rmsnorm_forward_cuda(x, weight, out));
}

void blt_rmsnorm_backward(const blt_tensor* grad_out, const blt_tensor* x, const blt_tensor* weight, blt_tensor* grad_x, blt_tensor* grad_weight){
    BLT_DISPATCH(x, blt_rmsnorm_backward_cpu(grad_out, x, weight, grad_x, grad_weight),
                 blt_rmsnorm_backward_cuda(grad_out, x, weight, grad_x, grad_weight));
}



// ------------------
// LINALG

void blt_matmul(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    BLT_DISPATCH(a, blt_matmul_cpu(a, b, out), blt_matmul_cuda(a, b, out));
}

void blt_matmul_backward(const blt_tensor* a, const blt_tensor* b, const blt_tensor* grad_out,
                        blt_tensor* grad_a, blt_tensor* grad_b){
    BLT_DISPATCH(a, blt_matmul_backward_cpu(a, b, grad_out, grad_a, grad_b),
                 blt_matmul_backward_cuda(a, b, grad_out, grad_a, grad_b));
}

// ------------------
// VECMATH (backend-keyed raw-pointer utilities)

float blt_vec_dot(blt_backend backend, const float* a, const float* b, size_t n) {
    BLT_DISPATCH_BACKEND_RET(backend,
        blt_vec_dot_cpu(backend, a, b, n),
        blt_vec_dot_cuda(backend, a, b, n));
}

void blt_softmax_masked_row_inplace(blt_backend backend, float* row, size_t row_len,
                                    size_t row_idx, bool is_causal,
                                    const float* mask_row, float scale) {
    BLT_DISPATCH_BACKEND(backend,
        blt_softmax_masked_row_inplace_cpu(backend, row, row_len, row_idx, is_causal, mask_row, scale),
        blt_softmax_masked_row_inplace_cuda(backend, row, row_len, row_idx, is_causal ? 1 : 0, mask_row, scale));
}

// ------------------
// OPTIMIZERS

void blt_sgd_step(blt_tensor* param, const blt_tensor* grad, float lr) {
    BLT_DISPATCH(param, blt_sgd_step_cpu(param, grad, lr), blt_sgd_step_cuda(param, grad, lr));
}

void blt_adamw_step(blt_tensor* param, const blt_tensor* grad,
                    blt_tensor* exp_avg, blt_tensor* exp_avg_sq,
                    const blt_adamw_config* config) {
    BLT_DISPATCH(param,
        blt_adamw_step_cpu(param, grad, exp_avg, exp_avg_sq, config),
        blt_adamw_step_cuda(param, grad, exp_avg, exp_avg_sq, config));
}

// ------------------
// MASK BUILDER (dispatch keyed on the arena that will own the mask)

void blt_build_attention_mask(const blt_mask_config* config, blt_tensor* out_mask, blt_arena* arena) {
    BLT_DISPATCH_BACKEND((blt_backend)(arena ? arena->backend : BLT_BACKEND_CPU),
        blt_build_attention_mask_cpu(config, out_mask, arena),
        blt_build_attention_mask_cuda(config, out_mask, arena));
}

void blt_build_block_diffusion_mask(const blt_block_diffusion_config* config,
                                    blt_tensor* out_mask, blt_arena* arena) {
    BLT_DISPATCH_BACKEND((blt_backend)(arena ? arena->backend : BLT_BACKEND_CPU),
        blt_build_block_diffusion_mask_cpu(config, out_mask, arena),
        blt_build_block_diffusion_mask_cuda(config, out_mask, arena));
}

// ------------------
// PATCH POOL

void blt_patch_pool_forward(const blt_tensor* byte_hidden, const blt_patch_info* patches,
                            size_t num_patches, blt_patch_pool_type pool_type, blt_tensor* out) {
    BLT_DISPATCH(byte_hidden,
        blt_patch_pool_forward_cpu(byte_hidden, patches, num_patches, pool_type, out),
        blt_patch_pool_forward_cuda(byte_hidden, patches, num_patches, pool_type, out));
}

void blt_patch_pool_backward(const blt_tensor* grad_out, const blt_tensor* byte_hidden,
                             const blt_patch_info* patches, size_t num_patches,
                             blt_patch_pool_type pool_type, blt_tensor* grad_byte_hidden) {
    BLT_DISPATCH(grad_out,
        blt_patch_pool_backward_cpu(grad_out, byte_hidden, patches, num_patches, pool_type, grad_byte_hidden),
        blt_patch_pool_backward_cuda(grad_out, byte_hidden, patches, num_patches, pool_type, grad_byte_hidden));
}
