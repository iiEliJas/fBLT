# FBLT API Reference

This is a brief reference for the Phase 0 API surface in the include and src trees.

## Core types and constants

### blt/core/dtype.h
- `blt_dtype` enum
  - `BLT_DTYPE_FP32` = 0: 32-bit floating-point tensor element type.
  - `BLT_DTYPE_BF16` = 1: brain-float16 tensor element type.
  - `BLT_DTYPE_INT32` = 2: 32-bit integer tensor element type.
  - `BLT_DTYPE_UINT8` = 3: 8-bit unsigned integer tensor element type.
- `blt_dtype_sizeof(dtype)`
  - Input: a `blt_dtype` value.
  - Output: the size in bytes of one element of that dtype.

### blt/core/tensor.h
- `BLT_MAX_NDIM` macro
  - Maximum supported tensor rank (currently 4).
- `blt_backend` enum
  - `BLT_BACKEND_CPU` = 0: CPU backend.
  - `BLT_BACKEND_CUDA` = 1: CUDA backend placeholder.
- `blt_tensor` struct
  - Fields:
    - `void* data`: backing storage for tensor values.
    - `size_t shape[BLT_MAX_NDIM]`: shape dimensions.
    - `size_t strides[BLT_MAX_NDIM]`: row-major strides.
    - `size_t ndim`: rank of the tensor.
    - `size_t numel`: total number of elements.
    - `blt_dtype dtype`: element type.
    - `blt_backend backend`: backend for the tensor.
    - `bool is_view`: whether the tensor is a view rather than owned storage.
- `blt_tensor_compute_numel(shape, ndim)`
  - Input: shape array and rank.
  - Output: total number of elements.
- `blt_tensor_compute_row_major_strides(shape, ndim, strides_out)`
  - Input: shape array, rank, and output stride buffer.
  - Output: fills `strides_out` with row-major strides.
- `blt_tensor_bytes(t)`
  - Input: pointer to a tensor.
  - Output: total byte size of the tensor storage.

### blt/core/backend.h
- `BLT_FATAL(msg)` macro
  - Input: an error message string.
  - Behavior: prints a fatal error to stderr and exits the program.
- `BLT_WARN(msg)` macro
  - Input: a warning message string.
  - Behavior: prints a warning to stderr.
- `BLT_REQUIRE(cond, msg)` macro
  - Input: a condition and a message string
  - Behavior: checks the condition and does a BLT_FATAL call if cond is false.
- `blt_check_2d_fp32(tensor, dim0, dim1, msg)`
  - Input: blt_tensor to check and dimensions and message
  - Behaviour: Validates a 2D FP32 tensor. Pass 0 for dim0/dim1 to skip that dimensions check.
- `blt_check_1d_fp32(tensor, dim0, msg)`
  - Input: blt_tensor to check and dimensions and message string
  - Behaviour: Validates a 1D FP32 tensor of exact length dim0.
- `blt_check_elementwise_fp32(a, b, msg)`
  - Input: two tensors and a message string
  - Behaviour: Validates that two tensors are FP32 and elementwise-compatible, regardless of rank

### blt/core/allocator.h
- `blt_arena` struct
  - Fields:
    - `void* buffer`: backing memory for the arena.
    - `size_t capacity`: total capacity in bytes.
    - `size_t offset`: current allocation offset.
    - `blt_backend backend`: backend associated with the arena.
- `blt_arena_create(capacity_bytes, backend)`
  - Input: arena capacity and backend.
  - Output: allocated arena handle, or aborts on failure.
- `blt_arena_destroy(arena)`
  - Input: arena handle.
  - Output: frees arena memory and associated buffer.
- `blt_arena_reset(arena)`
  - Input: arena handle.
  - Output: resets allocation offset to zero.
- `blt_arena_alloc(arena, bytes, alignment)`
  - Input: arena, allocation size, and alignment.
  - Output: pointer to aligned memory inside the arena, or aborts on failure.
- `blt_tensor_create(arena, shape, ndim, dtype)`
  - Input: arena for storage, shape, rank, and element type.
  - Output: a zero-initialized tensor with allocated storage.


------------------------------------------------------------------------------------------------------------
## Model APIs


### blt/models/attention.h
- `blt_attention_config` struct
  - Fields:
    - `size_t embed_dim`: total embedding dimension.
    - `size_t num_heads`: number of attention heads.
    - `size_t head_dim`: dimension per head; if zero, it may be inferred from `embed_dim / num_heads`.
    - `bool is_causal`: whether causal masking should be applied.
- `blt_multihead_attention(input, weight_qkv, weight_proj, output, config, arena)`
  - Input: input sequence tensor, QKV projection weights, output projection weights, output tensor, and attention config.
  - Output: writes the multi-head self-attention result into `output`.

### blt/models/transformer.h
- `blt_norm_type` enum
  - `BLT_NORM_LAYERNORM`: classic layer normalization with weight and bias.
  - `BLT_NORM_RMSNORM`: RMS normalization with weight only.
- `blt_activation_type` enum
  - `BLT_ACTIVATION_GELU`: standard GELU activation.
  - `BLT_ACTIVATION_SWIGLU`: SwiGLU activation with a gated projection.
- `blt_transformer_config` struct
  - Fields:
    - `blt_attention_config attn_config`: attention block configuration.
    - `size_t hidden_dim`: hidden width of the FFN intermediate projection.
    - `float layer_norm_eps`: epsilon for layer normalization; only used when `norm_type == BLT_NORM_LAYERNORM`.
    - `blt_norm_type norm_type`: normalization type for both attention and FFN pre-norms.
    - `blt_activation_type activation_type`: activation type used inside the FFN.
- `blt_transformer_weights` struct
  - Fields:
    - `const blt_tensor* norm1_weight`: first normalization weight.
    - `const blt_tensor* norm1_bias`: first normalization bias; NULL when using RMSNorm.
    - `const blt_tensor* attn_qkv_w`: QKV projection weights for attention.
    - `const blt_tensor* attn_proj_w`: output projection weights for attention.
    - `const blt_tensor* norm2_weight`: second normalization weight.
    - `const blt_tensor* norm2_bias`: second normalization bias; NULL when using RMSNorm.
    - `const blt_tensor* ffn_up_w`: FFN up-projection weights of shape `[embed_dim, hidden_dim]`.
    - `const blt_tensor* ffn_gate_w`: FFN gate projection weights of shape `[embed_dim, hidden_dim]`; required only when `activation_type == BLT_ACTIVATION_SWIGLU`.
    - `const blt_tensor* ffn_down_w`: FFN down-projection weights of shape `[hidden_dim, embed_dim]`.
- `blt_transformer_forward(input, weights, output, config, arena)`
  - Input: input sequence tensor, transformer weights, output tensor, transformer config, and scratch arena.
  - Output: writes a single transformer block forward pass result into `output`.
  - Behavior: performs pre-norm self-attention with residual, followed by pre-norm FFN with residual. The `arena` is used for intermediate tensors and is not reset by the function.

### blt/models/entropy.h
- `blt_entropy_config` struct
  - Fields:
    - `float threshold`: entropy threshold used to trigger a new patch.
    - `size_t vocab_size`: size of the probability distribution vocabulary.
    - `bool use_log2`: whether entropy should use base-2 logarithms.
- `blt_compute_entropy(probs, entropy_out, config)`
  - Input: 2D probability tensor and entropy output tensor.
  - Output: writes per-row entropy values into `entropy_out`.

### blt/models/patcher.h
- `blt_patch_info` struct
  - Fields:
    - `size_t start_idx`: starting index of the patch in the entropy sequence.
    - `size_t length`: number of elements in the patch.
    - `float peak_entropy`: maximum entropy value observed in the patch.
- `blt_segment_patches(entropy, patches_out, max_patches, config)`
  - Input: 1D entropy tensor, output patch buffer, maximum patch count, and entropy config.
  - Output: returns the number of produced patches and fills `patches_out` with patch metadata.


------------------------------------------------------------------------------------------------------------
## Operator APIs

### blt/ops/elementwise.h
- `blt_add(a, b, out)`
  - Input: two input tensors and an output tensor.
  - Output: writes elementwise `a + b` into `out`.
- `blt_mul(a, b, out)`
  - Input: two input tensors and an output tensor.
  - Output: writes elementwise `a * b` into `out`.
- `blt_scale(t, scalar)`
  - Input: one tensor to modify and a scalar float.
  - Output: multiplies each element of t with the scalar.

### blt/ops/matmul.h
- `blt_matmul(a, b, out)`
  - Input: two 2D input tensors and an output tensor.
  - Output: writes the matrix product into `out`.

### blt/ops/softmax.h
- `blt_softmax(in, out)`
  - Input: input tensor and output tensor.
  - Output: writes the softmax result into `out`.

### blt/ops/rope.h
- `blt_rope_precompute(size_t max_seq_len, const blt_rope_config* config, blt_tensor* cos_out, blt_tensor* sin_out)`
  - Input: maximum sequence length, rope configuration, and output tensors for cosine and sine.
  - Output: writes the precomputed sin and cos into cos_out, sin_out
  - Behaviour: Precomputes cos/sin tables for positions [0, max_seq_len) and head_dim/2 frequency bands

### blt/ops/layernorm.h
- `blt_layernorm_forward(x, weight, bias, out, eps)`
  - Input: 2D input tensor `[seq_len, embed_dim]`, per-channel weight and bias tensors of length `embed_dim`, output tensor, and epsilon.
  - Output: writes row-wise layer normalization over the last dimension into `out`.
- `blt_layernorm_backward(grad_out, x, weight, grad_x, grad_weight, grad_bias, eps)`
  - Input: gradient w.r.t. output, original input, weight tensor, output gradient tensors, and epsilon.
  - Output: computes backward gradients for `x`, `weight`, and `bias`.

### blt/ops/rmsnorm.h
- `blt_rmsnorm_forward(x, weight, out)`
  - Input: 2D input tensor `[seq_len, embed_dim]`, per-channel weight tensor of length `embed_dim`, and output tensor.
  - Output: writes row-wise RMS normalization into `out`.
- `blt_rmsnorm_backward(grad_out, x, weight, grad_x, grad_weight)`
  - Input: gradient w.r.t. output, original input, weight tensor, and output gradient tensors.
  - Output: computes backward gradients for `x` and `weight`.

### blt/ops/swiglu.h
- `blt_swiglu_forward(gate, up, out)`
  - Input: gate tensor, up tensor, and output tensor, all with matching element counts.
  - Output: writes elementwise `silu(gate) * up` into `out`.
- `blt_swiglu_backward(grad_out, gate, up, grad_gate, grad_up)`
  - Input: gradient w.r.t. output, gate tensor, up tensor, and gradient output tensors.
  - Output: computes backward gradients for `gate` and `up`.

### blt/ops/vecmath.h
- `blt_vec_dot(a, b, n)`
  - Input: two input data floats and size_t n.
  - Output: float dot product.
  - Behavior: computes the dot product in length n.

### blt/ops/gelu.h
- `blt_gelu_forward(x, out)`
  - Input: tensor input and tensor output.
  - Output: writes elementwise gelu result into out.
- `blt_gelu_backward(grad_out, x, grad_x)`
  - Not yet implemented.



------------------------------------------------------------------------------------------------------------
## Implementations

### src/core/allocator.c
- `blt_arena_create(...)`
  - Allocates and initializes an arena, rejects CUDA backends in Phase 0, and allocates aligned storage.
- `blt_arena_destroy(...)`
  - Frees the arena buffer and struct.
- `blt_arena_reset(...)`
  - Resets the arena offset to the beginning.
- `blt_arena_alloc(...)`
  - Allocates aligned bytes from the arena and advances the offset.
- `blt_tensor_create(...)`
  - Builds a tensor descriptor, computes shape/strides/numel, allocates storage, and zero-fills it.

### src/core/tensor.c
- `blt_tensor_compute_numel(...)`
  - Multiplies all shape dimensions to compute element count.
- `blt_tensor_compute_row_major_strides(...)`
  - Computes contiguous row-major strides for the given shape.
- `blt_tensor_bytes(...)`
  - Returns the byte size of the tensor payload.

### src/core/backend.c
- `blt_add(...)`
  - Dispatches to the CPU add implementation; raises a fatal error for CUDA unless CUDA support is enabled.
- `blt_mul(...)`
  - Dispatches to the CPU multiply implementation; raises a fatal error for CUDA unless CUDA support is enabled.
- `blt_matmul(...)`
  - Dispatches to the CPU matmul implementation; raises a fatal error for CUDA unless CUDA support is enabled.
- `blt_softmax(...)`
  - Dispatches to the CPU softmax implementation; raises a fatal error for CUDA unless CUDA support is enabled.

### src/models/attention.c
- `blt_multihead_attention(...)`
  - Validates tensor dtypes, ranks, shapes, and config values.
  - Builds a QKV projection, computes attention scores with optional causal masking, and applies the output projection.

### src/models/entropy.c
- `blt_compute_entropy(...)`
  - Validates that the input is a 2D probability tensor and that the output matches the expected row count.
  - Computes entropy per probability distribution using either `log2` or `ln` based on the config.

### src/models/patcher.c
- `blt_segment_patches(...)`
  - Validates that the entropy input is 1D and non-empty.
  - Segments the sequence into patches when entropy values cross the configured threshold and records each patch's start, length, and peak entropy.

### src/backend_cpu/elementwise_cpu.c
- `blt_add_cpu(a, b, out)`
  - Input: two FP32 tensors of equal element count and an FP32 output tensor.
  - Output: writes per-element addition into `out`.
- `blt_mul_cpu(a, b, out)`
  - Input: two FP32 tensors of equal element count and an FP32 output tensor.
  - Output: writes per-element multiplication into `out`.
- `validate_same_shape_and_dtype(...)` (static helper)
  - Checks that the operands and output have matching element counts and FP32 dtype.
- `blt_gelu_forward(x, out)`
  - Behaviour: computes elementwise GELU (tanh approximation): out = 0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))). 
  - x and out just need matching element count (any rank).
- `blt_gelu_backward(grad_out, x, grad_x)`
  - Not yet implemented.

### src/backend_cpu/linalg_cpu.c
- `blt_matmul_cpu(a, b, out)`
  - Input: two 2D FP32 tensors and an FP32 output tensor.
  - Output: writes a matrix multiplication result into `out`.
  - Requires: `a.shape[1] == b.shape[0]` and `out` shape must match the product dimensions.

### src/backend_cpu/reductions_cpu.c
- `blt_softmax_cpu(in, out)`
  - Input: FP32 input tensor and FP32 output tensor with matching element count.
  - Output: writes the softmax values into `out`.
  - Behavior: computes softmax across the last dimension of the tensor.