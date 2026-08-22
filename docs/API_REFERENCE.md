# FBLT API Reference

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
- `blt_tensor_compute_numel(const size_t* shape, size_t ndim)`
  - Input: shape array and rank.
  - Output: total number of elements.
- `blt_tensor_compute_row_major_strides(const size_t* shape, size_t ndim, size_t* strides_out)`
  - Input: shape array, rank, and output stride buffer.
  - Output: fills `strides_out` with row-major strides.
- `blt_tensor_bytes(const blt_tensor* t)`
  - Input: pointer to a tensor.
  - Output: total byte size of the tensor storage.
- `view_1d(blt_tensor* view, void* data, size_t len, blt_dtype dtype, blt_backend backend)`
  - Input: view tensor pointer, data pointer, element count, dtype, and backend.
  - Output: initializes `view` as a 1D view over the provided data.
- `view_1d_offset(blt_tensor* view, const blt_tensor* src, size_t offset_elems, size_t len)`
  - Input: view tensor pointer, source tensor, element offset into the source's data, and element count.
  - Output: initializes `view` as a 1D view starting `offset_elems` elements into `src`'s data; dtype and backend are inherited from `src`.
  - Behavior: convenience for shifted views (e.g. next-byte loss targets).
- `blt_tensor_view_2d(blt_tensor* t, void* data, size_t rows, size_t cols, blt_backend backend)`
  - Input: tensor pointer, data pointer, row count, column count, and backend.
  - Output: initializes `t` as a 2D view (FP32) over the provided data.
- `blt_tensor_view_3d(blt_tensor* t, void* data, size_t d0, size_t d1, size_t d2, blt_backend backend)`
  - Input: tensor pointer, data pointer, three dimension sizes, and backend.
  - Output: initializes `t` as a 3D view (FP32) over the provided data.
- `zero_tensor(blt_tensor* t)`
  - Input: tensor pointer.
  - Behavior: fills the tensor's storage with zeros. Note: tensors created via `blt_tensor_create` are already zero-initialized.


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
- `blt_check_nd_fp32(const blt_tensor* t, size_t ndim, const size_t* dims, const char* msg)`
  - Input: blt_tensor to check, number of dimensions, dimension array, and message string
  - Behaviour: Validates an N-dimensional FP32 tensor, including a non-NULL data pointer. Pass 0 for any dimension in `dims` to skip that dimension's check.
- `blt_check_elementwise_fp32(const blt_tensor* a, const blt_tensor* b, const char* msg)`
  - Input: two tensors and a message string
  - Behaviour: Validates that two tensors are FP32 and elementwise-compatible, regardless of rank

### blt/core/allocator.h
- `blt_arena` struct
  - Fields:
    - `void* buffer`: backing memory for the arena.
    - `size_t capacity`: total capacity in bytes.
    - `size_t offset`: current allocation offset.
    - `blt_backend backend`: backend associated with the arena.
- `blt_arena_create(size_t capacity_bytes, blt_backend backend)`
  - Input: arena capacity and backend.
  - Output: allocated arena handle, or aborts on failure.
- `blt_arena_destroy(blt_arena* arena)`
  - Input: arena handle.
  - Output: frees arena memory and associated buffer.
- `blt_arena_reset(blt_arena* arena)`
  - Input: arena handle.
  - Output: resets allocation offset to zero.
- `blt_arena_alloc(blt_arena* arena, size_t bytes, size_t alignment)`
  - Input: arena, allocation size, and alignment.
  - Output: pointer to aligned memory inside the arena, or aborts on failure.
- `blt_tensor_create(blt_arena* arena, const size_t* shape, size_t ndim, blt_dtype dtype)`
  - Input: arena for storage, shape, rank, and element type.
  - Output: blt_tensor - a zero-initialized tensor with allocated storage.



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
- `blt_matmul(const blt_tensor* a, const blt_tensor* b, blt_tensor* out)`
  - Input: two 2D input tensors and an output tensor.
  - Output: writes the matrix product into `out`.
- `blt_matmul_backward(const blt_tensor* a, const blt_tensor* b, const blt_tensor* grad_out, blt_tensor* grad_a, blt_tensor* grad_b)`
  - Input: two 2D input tensors, gradient output, and output gradient tensors.
  - Output: computes backward gradients for `a` and `b`.

### blt/ops/softmax.h
- `blt_softmax(in, out)`
  - Input: input tensor and output tensor.
  - Output: writes the softmax result into `out`.
- `blt_softmax_backward(const blt_tensor* grad_out, const blt_tensor* softmax_out, blt_tensor* grad_in)`
  - Input: gradient output, softmax output, and input gradient tensor.
  - Output: writes the gradient respect to the input into `grad_in` using the chain rule.

### blt/ops/cross_entropy.h
- `blt_cross_entropy_forward(logits, targets, loss_out)`
  - Input: `logits` [seq_len, vocab_size] FP32 (not normalized) and `targets` [seq_len] UINT8 or INT32 input tensors, `loss_out` output tensor
  - Output: writes scalar mean cross-entropy over seq_len into loss_out
- `blt_cross_entropy_backward(logits, targets, grad_logits)`
  - Input: `logits`, `targets` input tensors and `grad_logits` as output tensor
  - Output: Writes dL/dlogits into `grad_logits` [seq_len, vocab_size]

### blt/ops/rope.h
- `blt_rope_config` struct
  - Fields:
    - `float theta`: base for rotary position embedding (e.g., 500000.0f).
    - `size_t head_dim`: dimension per attention head (must be even).
- `blt_rope_precompute(size_t max_seq_len, const blt_rope_config* config, blt_tensor* cos_out, blt_tensor* sin_out)`
  - Input: maximum sequence length, rope configuration, and output tensors for cosine and sine.
  - Output: writes the precomputed sin and cos into cos_out, sin_out
  - Behaviour: Precomputes cos/sin tables for positions [0, max_seq_len) and head_dim/2 frequency bands
- `blt_rope_apply(const blt_tensor* x, const blt_tensor* cos, const blt_tensor* sin, blt_tensor* out)`
  - Input: input tensor x, precomputed cos and sin tables, and output tensor.
  - Output: applies rotary embedding in place to x using the precomputed cos/sin tables.
  - Behaviour: reads x [seq_len, num_heads, head_dim], writes rotated result into out (same shape)
- `blt_rope_apply_backward(const blt_tensor* grad_out, const blt_tensor* cos, const blt_tensor* sin, blt_tensor* grad_in)`
  - Input: gradient output, precomputed cos and sin tables, and input gradient tensor.
  - Output: computes the backward gradient for x using the chain rule and writes into grad_in.

### blt/ops/layernorm.h
- `blt_layernorm_forward(const blt_tensor* x, const blt_tensor* weight, const blt_tensor* bias, blt_tensor* out, float eps)`
  - Input: 2D input tensor `[seq_len, embed_dim]`, per-channel weight and bias tensors of length `embed_dim`, output tensor, and epsilon.
  - Output: writes row-wise layer normalization over the last dimension into `out`.
- `blt_layernorm_backward(const blt_tensor* grad_out, const blt_tensor* x, const blt_tensor* weight, blt_tensor* grad_x, blt_tensor* grad_weight, blt_tensor* grad_bias, float eps))`
  - Input: gradient respect to the output, original input, weight tensor, output gradient tensors, and epsilon.
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
  - Input: gradient respect to the output, gate tensor, up tensor, and gradient output tensors.
  - Output: computes backward gradients for `gate` and `up`.

### blt/ops/vecmath.h
- `blt_vec_dot(a, b, n)`
  - Input: two input data floats and size_t n.
  - Output: float dot product.
  - Behavior: computes the dot product in length n.
- `blt_softmax_masked_row_inplace(row, row_len, row_idx, is_causal, mask_row, scale)`
  - Input: contiguous FP32 row of length `row_len`, the row's index (for causal masking), `is_causal` flag, optional additive `mask_row` (same length; e.g. 0 or `-INFINITY` entries), and score `scale`.
  - Output: softmaxed row in place.
  - Behavior: numerically stable masked/causal row-softmax shared by self-attention and cross-attention. Masking takes precedence over causal masking when `mask_row != NULL`. Non-finite entries after masking contribute zero; a fully masked row yields all zeros. Raw-pointer utility, not a `blt_tensor` op.

### blt/ops/gelu.h
- `blt_gelu_forward(x, out)`
  - Input: tensor input and tensor output.
  - Output: writes elementwise gelu result into out.
- `blt_gelu_backward(grad_out, x, grad_x)`
  - Input: gradient respect to output, original input, and output gradient tensor.
  - Output: computes backward gradient for `x` and writes into `grad_x`.

### blt/ops/optim.h
- `blt_sgd_step(blt_tensor* param, const blt_tensor* grad, float lr)`
  - Input: `param` tensor to update in place, `grad` tensor (same shape/dtype as `param`), and learning rate `lr`.
  - Output: none; updates `param` in place.
  - Behavior: elementwise `param -= lr * grad`. CPU-only, stateless (no momentum/second-moment buffers) — the minimal Phase 1a optimizer. Requires `param` and `grad` to be FP32 and elementwise-compatible (validated via `blt_check_elementwise_fp32`) and `param->backend == BLT_BACKEND_CPU`.

## blt/ops/mask_builder.h
- `blt_mask_config` struct
  - Fields:
      - `size_t seq_len_q`: Query sequence length
      - `size_t seq_len_kv`: Key/Value sequence length
      - `size_t sliding_window`: 0 for full causal attention, otherwise sliding window size
      - `const size_t* doc_boundaries`: Array of indices where new documents start
      - `size_t num_docs`: Number of documents
      - `const size_t* query_group_ids`: Group ID for each query position
      - `const size_t* kv_group_ids`: Group ID for each KV position
      - `bool bidirectional_within_group`: If true, attend to all positions in group; if false, causal within group
      - `bool is_causal`: Apply causal mask
- `blt_build_attention_mask(config, out_mask, arena)`
    - Input: `blt_mask_config` struct defining attention constraints, output tensor, and arena for storage
    - Output: Writes a 2D FP32 mask tensor of shape `[seq_len_q, seq_len_kv]` to `out_mask`. Uses `0` for allowed attention and `-INFINITY` for masked attention

### blt/ops/patch_pool.h
- `blt_patch_pool_type` enum
  - `BLT_POOL_MEAN` = 0: Mean pooling; averages byte representations within each patch boundary.
  - `BLT_POOL_MAX` = 1: Max pooling; selects per-channel maximum byte representation within each patch.
- `blt_patch_pool_forward(byte_hidden, patches, num_patches, pool_type, out)`
  - Input: `byte_hidden` tensor `[seq_len, embed_dim]` (FP32), `patches` metadata array `[num_patches]`, count `num_patches`, `pool_type` strategy, and target output tensor `out`.
  - Output: Writes pooled patch representations into `out` `[num_patches, embed_dim]` (FP32).
  - Behavior: Computes row-wise patch features (mean or per-channel max) over contiguous byte spans defined by `patches`. Validates that patch spans cover valid ranges in `seq_len`.
- `blt_patch_pool_backward(grad_out, byte_hidden, patches, num_patches, pool_type, grad_byte_hidden)`
  - Input: `grad_out` `[num_patches, embed_dim]`, original `byte_hidden` `[seq_len, embed_dim]`, `patches` metadata, `num_patches`, `pool_type`, and `grad_byte_hidden` `[seq_len, embed_dim]`.
  - Output: Accumulates input gradients into `grad_byte_hidden` (must be zero-initialized prior to call).
  - Behavior: For `MEAN`, scatters `grad_out[j] / patch.length` across all constituent bytes in patch $j$. For `MAX`, recomputes the per-channel argmax using `byte_hidden` and routes the total channel gradient exclusively to the argmax byte position.
- `blt_patch_build_group_ids(patches, num_patches, seq_len, query_group_ids_out, kv_group_ids_out)`
  - Input: `patches` metadata array `[num_patches]`, `num_patches`, `seq_len` (total bytes), caller-allocated `query_group_ids_out` `[num_patches]`, and `kv_group_ids_out` `[seq_len]`.
  - Output: Fills `query_group_ids_out` with query patch indices ($0 \dots \text{num\_patches}-1$) and `kv_group_ids_out` with parent patch indices for each byte position.
  - Behavior: Maps queries and key/value positions into patch groups for block-diagonal cross-attention masks. Asserts that input patches form a contiguous tiling over `[0, seq_len)`.
  - `blt_patch_expand_group_ids(group_ids_in, n, k, group_ids_out)`
    - Input: `group_ids_in` array `[n]`, expansion factor `k`, and output array `group_ids_out` `[n * k]`.
    - Output: Fills `group_ids_out` by repeating each input group ID `k` times consecutively.
    - Behavior: Expands each input group ID into `k` consecutive output group IDs.

------------------------------------------------------------------------------------------------------------
## Model APIs

### blt/models/entropy.h
- `blt_entropy_config` struct
  - Fields:
    - `size_t vocab_size`: size of the probability distribution vocabulary.
    - `bool use_log2`: whether entropy should use base-2 logarithms.
- `blt_compute_entropy(probs, entropy_out, config)`
  - Input: 2D probability tensor and entropy output tensor.
  - Output: writes per-row entropy values into `entropy_out`.

### blt/models/patcher.h
- `blt_patch_rule` enum
  - `BLT_PATCH_RULE_GLOBAL`: use only the global threshold rule
  - `BLT_PATCH_RULE_MONOTONIC`: use only the monotonic threshold rule
  - `BLT_PATCH_RULE_BOTH`: use both rules
- `blt_patch_info` struct
  - Fields:
    - `size_t start_idx`: starting index of the patch in the entropy sequence.
    - `size_t length`: number of elements in the patch.
    - `float peak_entropy`: maximum entropy value observed in the patch.
- `blt_patcher_config` struct
  - Fields:
    - `float threshold_global`: global patch rule value (e.g. 1.34f)
    - `float threshold_monotonic`: monotonic rule value (e.g. 0.5f)
    - `size_t max_patch_length`: maximum number of bytes per patch
    - `blt_patch_rule rule`: which rule to use
    - `bool reset_on_newline`: starts a new patch after `\n`    
- `blt_segment_patches(entropy, bytes, patches_out, max_patches, config)`
  - Input: 1D entropy tensor, uint8_t bytes array, output patch buffer, maximum patch count (must be >= 1), and patcher config
  - Output: returns the number of produced patches and fills `patches_out` with patch metadata

## blt/models/byte_embedding.h
- `blt_byte_embedding` struct
  - Fields:
    - `size_t vocab_size`: embedding vocabulary - 256 or 257 (for the MASK token)
    - `blt_tensor weight`: embedding table of shape `[256, embed_dim]` in FP32
    - `size_t embed_dim`: dimension of the embedding vectors
- `blt_byte_embedding_create(arena, vocab_size, embed_dim)`
  - Input: arena for storage, size_t vocab size and embedding dimension
  - Output: allocated and zero-initialized embedding table; caller fills `weight->data`
- `blt_byte_embedding_forward(emb, bytes_in, out)`
  - Input: embedding struct, 1D UINT8 tensor of raw byte values `[seq_len]`, and output tensor
  - Output: writes the corresponding 2D FP32 embeddings `[seq_len, embed_dim]` into `out`
- `blt_byte_embedding_backward(emb, bytes_in, grad_out, grad_weight)`
  - Input: embedding struct, 1D UINT8 tensor of raw byte values `[seq_len]`, gradient output tensor, and gradient weight tensor
  - Output: scatter-adds `grad_out` rows into `grad_weight` at the row indexed by the corresponding input byte value. `grad_weight` must be zeroed by the caller before accumulating across a batch

### blt/models/attention.h
- `blt_attention_config` struct
  - Fields:
    - `size_t embed_dim`: total embedding dimension
    - `size_t num_heads`: number of attention heads
    - `size_t head_dim`: dimension per head; if zero, it may be inferred from `embed_dim / num_heads`
    - `bool is_causal`: whether causal masking should be applied
    - `bool use_rope`: use rotary position embedding
    - `float rope_theta`: base for rotary position embedding
    - `const blt_tensor* rope_cos_cache`: precomputed cos table for RoPE (optional)
    - `const blt_tensor* rope_sin_cache`: precomputed sin table for RoPE (optional)
    - `const blt_mask_config* mask_config`: config for masking attention weights (optional)
- `blt_multihead_attention(input, weight_qkv, weight_proj, output, config, arena)`
  - Input: input sequence tensor, QKV projection weights, output projection weights, output tensor, and attention config
  - Output: writes the multi-head self-attention result into `output`.
- `blt_multihead_attention_backward(input, weight_qkv, weight_proj, grad_out, grad_input, grad_weight_qkv, grad_weight_proj, config, arena)`
  - Input: input sequence tensor, QKV projection weights, output projection weights, gradient output tensor, gradient input tensor, gradient QKV weights tensor, gradient output projection weights tensor, and attention config.
  - Output: computes backward gradients for the attention block and writes into the provided gradient tensors.

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
- `blt_transformer_layer_storage` struct
  - Fields: `blt_tensor norm1_weight`, `attn_qkv_w`, `attn_proj_w`, `norm2_weight`, `ffn_up_w`, `ffn_gate_w`, `ffn_down_w` - owned storage for one transformer layer's weights.
- `blt_transformer_layer_grad` struct
  - Fields: `blt_tensor norm1_weight`, `attn_qkv_w`, `attn_proj_w`, `norm2_weight`, `ffn_up_w`, `ffn_gate_w`, `ffn_down_w` — gradient counterpart of `blt_transformer_layer_storage`.
- `blt_transformer_forward(input, weights, output, config, arena)`
  - Input: input sequence tensor, transformer weights, output tensor, transformer config, and scratch arena.
  - Output: writes a single transformer block forward pass result into `output`.
  - Behavior: performs pre-norm self-attention with residual, followed by pre-norm FFN with residual. The `arena` is used for intermediate tensors and is not reset by the function.

### blt/models/transformer_stack.h
- `blt_transformer_stack_config` struct
  - Fields:
    - `size_t num_layers`: number of stacked transformer blocks.
    - `size_t embed_dim`: transformer hidden width.
    - `size_t hidden_dim`: FFN intermediate width.
    - `size_t num_heads`: attention heads; `embed_dim` must be divisible by `num_heads`.
    - `size_t max_seq_len`: upper bound used to size the shared RoPE cache.
    - `float rope_theta`: RoPE base (e.g. `500000.0f`).
- `blt_transformer_stack` struct
  - Fields:
    - `blt_transformer_config layer_config`: shared per-layer config for every block in the stack.
    - `blt_transformer_layer_storage* layer_storage`: owned weight storage, `[num_layers]`.
    - `blt_transformer_weights* layer_weights`: const-pointer views into `layer_storage`, `[num_layers]` — what is actually passed to the layer forward pass.
    - `blt_tensor rope_cos_cache`: shared precomputed cosine RoPE cache of shape `[max_seq_len, head_dim/2]`.
    - `blt_tensor rope_sin_cache`: shared precomputed sine RoPE cache of shape `[max_seq_len, head_dim/2]`.
    - `size_t num_layers`, `embed_dim`, `hidden_dim`, `num_heads`, `head_dim`, `max_seq_len`: stack metadata.
- `blt_transformer_stack_grad` struct
  - Fields:
    - `blt_transformer_layer_grad* layer_grads`: gradient storage for each layer, `[num_layers]`.
- `blt_transformer_stack_init(blt_arena* arena, blt_transformer_stack* stack, const blt_transformer_stack_config* config)`
  - Input: scratch arena, stack to initialize, and stack config.
  - Output: allocates each layer's weight storage and precomputes the shared RoPE cache.
  - Behavior: caller fills the weight tensors afterward, using the same contract as `blt_tensor_create`.
- `blt_transformer_stack_grad_create(blt_arena* arena, const blt_transformer_stack* stack)`
  - Input: arena for storage and the stack to mirror.
  - Output: allocated, zero-initialized gradient struct matching the stack's shapes.
- `blt_transformer_stack_call_config(const blt_transformer_stack* stack, size_t seq_len, blt_tensor* cos_view, blt_tensor* sin_view)`
  - Input: stack, sequence length (must be in `[1, max_seq_len]` — the RoPE views index into the shared cache), and caller-owned RoPE views sized for `[seq_len, head_dim/2]`.
  - Output: returns a per-call `blt_transformer_config` whose RoPE cache pointers are overwritten to that sequence-length view.
  - Behavior: copies the shared template config and allows callers to attach custom masks (e.g., a block-causal document mask) on the returned `attn_config.mask_config`.
- `blt_transformer_stack_forward(const blt_transformer_stack* stack, const blt_tensor* x, const blt_transformer_config* call_cfg, size_t seq_len, blt_tensor* out, blt_arena* arena)`
  - Input: stack, input tensor `x`, per-call config, sequence length, output tensor `out`, and scratch arena.
  - Output: writes stacked transformer activations into `out` without storing the intermediate cache.
  - Behavior: applies the stack as a sequence of identical RMSNorm + SwiGLU transformer layers with RoPE on the input stream.
- `blt_transformer_stack_forward_cached(const blt_transformer_stack* stack, const blt_tensor* x, const blt_transformer_config* call_cfg, size_t seq_len, blt_arena* arena, blt_tensor* out)`
  - Input: stack, input, call config, sequence length, arena, and output tensor.
  - Output: returns a per-call cache object and writes the stacked forward result into `out`.
  - Behavior: the cache must be passed to `blt_transformer_stack_backward` to backpropagate through the same computation.
- `blt_transformer_stack_backward(const blt_transformer_stack* stack, const blt_transformer_stack_cache* cache, const blt_transformer_config* call_cfg, size_t seq_len, const blt_tensor* grad_out, blt_transformer_stack_grad* grad, blt_tensor* grad_x, blt_arena* arena)`
  - Input: stack, cached forward pass, call config, sequence length, output gradient, gradient accumulator, and scratch arena.
  - Output: writes per-layer weight gradients into `grad` and input gradients into `grad_x`.
  - Behavior: walks the cached stack in reverse layer order, propagating gradients through the shared transformer primitives in the same order they were computed during forward.

### blt/models/global_transformer.h
- `blt_global_transformer_config` struct
  - Fields:
    - `size_t embed_dim`: patch embedding width (`h_G`).
    - `size_t num_layers`: number of global transformer blocks (`l_G`).
    - `size_t hidden_dim`: FFN width for the global stack.
    - `size_t num_heads`: number of attention heads.
    - `float rope_theta`: RoPE base frequency.
    - `size_t max_seq_len`: maximum number of patches supported by the shared RoPE cache.
- `blt_global_transformer` struct
  - Fields:
    - `blt_global_transformer_config config`: parameter block for the model.
    - `blt_transformer_stack stack`: shared stack of `num_layers` transformer blocks with RoPE.
- `blt_global_transformer_grad` struct
  - Fields:
    - `blt_transformer_stack_grad* stack_grad`: per-layer gradient storage, `[num_layers]`.
- `blt_global_transformer_create(blt_arena* arena, const blt_global_transformer_config* config)`
  - Input: arena for storage and global-transformer config.
  - Output: allocated model with zero-initialized weights and a shared RoPE cache sized to `max_seq_len`.
  - Behavior: caller fills weight data afterward; the same shared RoPE cache is reused across all layers.
- `blt_global_transformer_grad_create(blt_arena* arena, const blt_global_transformer* model)`
  - Input: arena and the model to mirror.
  - Output: allocates a gradient struct with the same shapes as the model.
- `blt_global_transformer_forward(const blt_global_transformer* model, const blt_tensor* patch_in, const size_t* doc_boundaries, size_t num_docs, blt_tensor* patch_out, blt_arena* arena)`
  - Input: model, input patch tensor `[num_patches, embed_dim]`, document boundary offsets `doc_boundaries` in patch indices, document count `num_docs`, output patch tensor `patch_out`, and scratch arena.
  - Output: writes contextualized patch representations `[num_patches, embed_dim]`.
  - Behavior: builds a block-causal patch mask (full causal at patch level, scoped by document boundaries) and runs the stack over the patch stream. This is the patch-level decoder-only transformer used after the local encoder.
- `blt_global_transformer_backward(const blt_global_transformer* model, const blt_tensor* patch_in, const size_t* doc_boundaries, size_t num_docs, const blt_tensor* grad_patch_out, blt_tensor* grad_patch_in, blt_global_transformer_grad* grad, blt_arena* arena)`
  - Input: model, patch inputs, document boundaries, output-gradient tensor, gradient accumulator, and scratch arena.
  - Output: writes gradients for the input patches and each stack layer into `grad`.
  - Behavior: recomputes the forward pass with per-layer intermediates cached, then backpropagates across layers in reverse order through the shared transformer stack.

### blt/models/entropy_lm.h
Assembly of already-implemented pieces (byte embedding, RoPE-enabled transformer stack, cross-entropy) into one callable "tiny causal byte LM": embedding → N transformer layers → LM head → shifted next-byte cross-entropy loss.
- `blt_entropy_lm_config` struct
  - Fields:
    - `size_t embed_dim`: transformer hidden width.
    - `size_t num_layers`: number of stacked transformer blocks.
    - `size_t hidden_dim`: FFN intermediate width.
    - `size_t num_heads`: attention heads; `embed_dim` must be divisible by `num_heads`.
    - `size_t max_seq_len`: upper bound used to size the shared RoPE cache.
    - `float rope_theta`: RoPE base (e.g. 500000.0f).
- `blt_entropy_lm` struct
  - Fields:
    - `blt_entropy_lm_config config`
    - `blt_tensor embedding_weight`: `[256, embed_dim]`.
    - `blt_tensor lm_head_weight`: `[embed_dim, 256]`.
    - `blt_transformer_stack stack`: shared stack of `num_layers` transformer blocks with RoPE.
- `blt_entropy_lm_grad` struct
  - Fields:
    - `blt_tensor embedding_grad`: `[256, embed_dim]`.
    - ` blt_transformer_stack_grad* stack_grad`: per-layer gradient storage, `[num_layers]`.
    - `blt_tensor lm_head_grad`: `[embed_dim, 256]`.
- `blt_entropy_lm_create(blt_arena* arena, const blt_entropy_lm_config* config)`
  - Input: arena for storage and model config.
  - Output: allocated model with every weight tensor zero-initialized, including the shared RoPE cache (which is precomputed here via `blt_rope_precompute`, sized to `max_seq_len`, and shared by pointer across every layer's `attn_config`). Caller fills weight data afterward.
- `blt_entropy_lm_grad_create(blt_arena* arena, const blt_entropy_lm* model)`
  - Input: arena for storage and the model to mirror.
  - Output: allocated, zero-initialized gradient struct matching `model`'s shapes. `blt_entropy_lm_backward` accumulates (scatter-adds, for the embedding table) or overwrites (every other weight) into this struct.
- `blt_entropy_lm_forward(const blt_entropy_lm* model, const blt_tensor* bytes_in, blt_tensor* logits_out, blt_tensor* loss_out, blt_arena* arena)`
  - Input: model, `bytes_in` `[seq_len]` UINT8 (`seq_len >= 2` and `<= config.max_seq_len`), caller-allocated `logits_out` `[seq_len, 256]` FP32, caller-allocated scalar `loss_out`, and scratch arena.
  - Output: writes the full `[seq_len, 256]` logits into `logits_out` and the shifted next-byte cross-entropy loss into `loss_out`.
  - Behavior: embedding → N transformer layers → matmul against `lm_head_weight` → logits → cross-entropy against shifted targets (`targets[t] = bytes_in[t+1]`, so the loss is computed over `seq_len - 1` positions). Builds a `seq_len`-sized view over the model's precomputed RoPE cache for this call, since `blt_multihead_attention` validates the cache shape as exactly `[seq_len, head_dim/2]`.
- `blt_entropy_lm_backward(const blt_entropy_lm* model, const blt_tensor* bytes_in, blt_entropy_lm_grad* grad_out, blt_arena* arena)`
  - Input: model, `bytes_in` (same as forward), caller-allocated `grad_out` (see `blt_entropy_lm_grad_create`), and scratch arena.
  - Output: writes gradients for every learnable weight into `grad_out`.
  - Behavior: recomputes the forward pass internally (caching each layer's intermediates, since `blt_transformer_forward` doesn't expose them — the same recompute pattern `blt_multihead_attention_backward` uses), then mirrors the forward call order in reverse: cross-entropy → LM head matmul → transformer layers (reverse) → embedding, calling each op's existing backward (`blt_rmsnorm_backward`, `blt_multihead_attention_backward`, `blt_swiglu_backward`, `blt_matmul_backward`, `blt_cross_entropy_backward`, `blt_byte_embedding_backward`) rather than introducing new math.

### blt/models/hash_ngram.h
- `BLT_MAX_NGRAM_SIZES` macro
  - Maximum number of different n-gram sizes supported (currently 6).
- `blt_hash_ngram_config` struct
  - Fields:
    - `size_t ngram_sizes[BLT_MAX_NGRAM_SIZES]`: Array of active n-gram sizes (e.g., {3, 4, 5, 6, 7, 8}).
    - `size_t num_ngram_sizes`: Number of active entries in `ngram_sizes`.
    - `size_t per_ngram_vocab`: Size of each hash embedding table (acts as modulus for the rolling hash).
    - `uint64_t hash_prime`: Base prime for the rolling polynomial hash.
    - `bool normalize`: If true, divides the output by `(num_ngram_sizes + 1)`.
    - `size_t embed_dim`: Dimensionality of the embeddings.
- `blt_hash_ngram_weights` struct
  - Fields:
    - `blt_tensor tables[BLT_MAX_NGRAM_SIZES]`: Array of tensors, each `[per_ngram_vocab, embed_dim]` FP32.
    - `size_t num_tables`: Number of active tables (matches `num_ngram_sizes`).
- `blt_rolling_hash_state` struct
  - Stateful rolling hash to compute polynomial hashes in O(1) per position.
  - Fields:
    - `uint64_t current_hash`: The hash value of the current window.
    - `uint64_t prime`: The base prime for the polynomial hash.
    - `uint64_t modulus`: The modulus (typically the vocab size).
    - `uint64_t prime_pow_n`: Precomputed `prime^n % modulus`.
    - `uint8_t window[8]`: Circular buffer of the last `n` bytes.
    - `size_t window_start`: Index of the oldest byte in the circular buffer.
    - `size_t n`: The n-gram size.
    - `size_t positions_seen`: How many bytes have been fed so far.
- `blt_rolling_hash_init(state, n, prime, modulus)`
  - Input: pointer to `blt_rolling_hash_state`, n-gram size `n`, base `prime`, and `modulus`.
  - Output: None. Initializes the state and precomputes `prime^n % modulus`.
  - Behavior: Validates that `n` is in `[1, 8]` and that the rolling update's largest intermediate (`(modulus-1)*prime + 255`) does not overflow `uint64`.
- `blt_rolling_hash_update(state, new_byte)`
  - Input: pointer to `blt_rolling_hash_state` and a `uint8_t` byte.
  - Output: Returns the hash of the current n-gram if `positions_seen >= n`. Returns `UINT64_MAX` if not enough bytes have been seen yet.
  - Behavior: Uses a circular buffer to track the last `n` bytes and updates the hash in O(1) time using the rolling polynomial method: `H_new = (H_old * prime + b_new - b_outgoing * prime^n) mod modulus`.
- `blt_hash_ngram_create(arena, config)`
  - Input: arena for storage and `blt_hash_ngram_config`.
  - Output: Returns an allocated `blt_hash_ngram_weights` struct.
  - Behavior: Allocates embedding tables and initializes them with small uniform random values in `[-0.02, 0.02]` to break symmetry.
- `blt_hash_ngram_forward(weights, config, bytes_in, byte_emb, out)`
  - Input: `weights`, `config`, `bytes_in` (1D UINT8 `[seq_len]`), `byte_emb` (2D FP32 `[seq_len, embed_dim]`).
  - Output: `out` (2D FP32 `[seq_len, embed_dim]`). Contains `byte_emb` + n-gram embeddings.
  - Behavior: For each position `i`, adds the hash table lookups for all active n-gram sizes to the base byte embedding. Positions `i < n-1` receive no contribution from the size-`n` table. If `normalize` is true, scales the final output by `1 / (num_ngram_sizes + 1)`.
- `blt_hash_ngram_backward(config, bytes_in, grad_out, grad_byte_emb, grad_tables)`
  - Input: `config`, `bytes_in`, `grad_out` (2D FP32 `[seq_len, embed_dim]`).
  - Output: `grad_byte_emb` (2D FP32 `[seq_len, embed_dim]`) and `grad_tables` (array of 2D FP32 `[per_ngram_vocab, embed_dim]`).
  - Behavior: Computes gradients w.r.t the base byte embeddings and the hash tables. Scales gradients by `1 / (num_ngram_sizes + 1)` if `normalize` is true. Scatter-adds gradients into `grad_tables` (which MUST be zero-initialized by the caller). To avoid aliasing issues if `grad_byte_emb` and `grad_out` point to the same memory, the scatter-add is executed strictly before overwriting `grad_byte_emb`.

### blt/models/cross_attention.h
- `blt_cross_attention_split_mode`enum
  - `BLT_CROSS_ATTN_NO_SPLIT=0`: k = 1, current behavior, unchanged
  - `BLT_CROSS_ATTN_SPLIT_QUERY`: query_in is `[n_q, patch_dim]`; kv_in is `[n_kv, embed_dim]` (encoder usage)
  - `BLT_CROSS_ATTN_SPLIT_KV`: kv_in is `[n_kv, patch_dim]`; query_in is `[n_q, embed_dim]` (decoder usage)
- `blt_cross_attention_config` struct
  - Fields:
    - `size_t embed_dim`: Shared hidden width ($h_E$) for query and key/value inputs.
    - `size_t patch_dim`: Width of the patch input; if 0, same as `embed_dim` (no splitting).
    - `blt_cross_attention_split_mode split_mode`: Which input is patch-dimension-wide.
    - `size_t num_heads`: Number of cross-attention heads ($U_E$).
    - `size_t head_dim`: Per-head dimension; if 0, inferred as `embed_dim / num_heads`.
    - `const blt_mask_config* mask_config`: REQUIRED pointer to block-diagonal patch mask config.
- `blt_cross_attention_weights` struct
  - Fields:
    - `const blt_tensor* weight_q`: Projection weights for Q of shape `[embed_dim, embed_dim]`.
    - `const blt_tensor* weight_k`: Projection weights for K of shape `[embed_dim, embed_dim]`.
    - `const blt_tensor* weight_v`: Projection weights for V of shape `[embed_dim, embed_dim]`.
    - `const blt_tensor* weight_proj`: Output projection weights of shape `[embed_dim, embed_dim]`.
- `blt_cross_attention_grad` struct
  - Fields:
    - `blt_tensor grad_weight_q`: Gradient tensor for Q projection of shape `[embed_dim, embed_dim]`.
    - `blt_tensor grad_weight_k`: Gradient tensor for K projection of shape `[embed_dim, embed_dim]`.
    - `blt_tensor grad_weight_v`: Gradient tensor for V projection of shape `[embed_dim, embed_dim]`.
    - `blt_tensor grad_weight_proj`: Gradient tensor for output projection of shape `[embed_dim, embed_dim]`.
- `blt_cross_attention_forward(query_in, kv_in, weights, output, config, arena)`
  - Input: `query_in` `[num_patches, embed_dim]` ($P_{l-1}$), `kv_in` `[seq_len, embed_dim]` ($h_l$), `weights`, output tensor `output`, `config` (with non-NULL `mask_config`), and scratch `arena`.
  - Output: Writes pre-residual cross-attention output into `output` `[num_patches, embed_dim]`.
  - Behavior: Projects queries from `query_in` and keys/values from `kv_in`, computes scaled dot-product attention per head using block-diagonal patch masking, concatenates heads, and applies output projection ($W_o$). Intermediates are allocated from `arena`.
- `blt_cross_attention_backward(query_in, kv_in, weights, grad_out, grad_query_in, grad_kv_in, grad_weights, config, arena)`
  - Input: Same `query_in`, `kv_in`, `weights`, `config`, and `arena` as forward, plus `grad_out` `[num_patches, embed_dim]` ($dL/d\text{Output}$).
  - Output: Writes input gradients into `grad_query_in` `[num_patches, embed_dim]`, `grad_kv_in` `[seq_len, embed_dim]`, and weight gradients into `grad_weights` (all overwritten).
  - Behavior: Recomputes forward intermediates ($Q$, $K$, $V$, per-head attention probabilities) into `arena`, then computes reverse-pass gradients for query inputs, key/value inputs, and projection weights via `blt_matmul_backward` and `blt_softmax_backward`. Note: the K and V input gradients are summed internally before being written to `grad_kv_in`, which is overwritten (not accumulated into).

### blt/models/local_common.h

Shared per-layer weight layout used by both the local encoder and local decoder. Each layer composes a cross-attention block (RMSNorm + QKV cross-attention) and a byte-level transformer block (causal self-attn + FFN); only the order of operations differs between the two models.

- `blt_local_layer_storage` struct
  - Fields: byte transformer block weights (`norm1_weight`, `attn_qkv_w`, `attn_proj_w`, `norm2_weight`, `ffn_up_w`, `ffn_gate_w`, `ffn_down_w`) and cross-attention block weights (`cross_norm_weight`, `cross_weight_q`, `cross_weight_k`, `cross_weight_v`, `cross_weight_proj`) — all owned storage.
  - Note: `blt_local_encoder_layer_storage` and `blt_local_decoder_layer_storage` are typedef aliases of this type; field access through either name is valid.
- `blt_local_layer_grad` struct
  - Fields: gradient counterpart mirroring `blt_local_layer_storage`. The per-model types `blt_local_encoder_layer_grad` and `blt_local_decoder_layer_grad` are typedef aliases of this type.
- `blt_local_layer_storage_alloc(arena, storage, embed_dim, hidden_dim)`
  - Behavior: allocates all tensors of a shared layer storage (zero-initialized).
- `blt_local_layer_grad_alloc(arena, grad, embed_dim, hidden_dim)`
  - Behavior: allocates all tensors of a shared layer grad (zero-initialized).
- `blt_local_cross_attn_fires(bool cross_attn_all_layers, size_t num_layers, size_t layer)`
  - Output: whether cross-attention fires on `layer` — every layer when `cross_attn_all_layers`, otherwise only after the final layer.
- `blt_local_byte_weights_view(const blt_local_layer_storage* s)`
  - Output: `blt_transformer_weights` view of the byte-transformer block for use with `blt_transformer_forward`/`blt_transformer_layer_forward_cached`.
- `blt_local_cross_weights_view(const blt_local_layer_storage* s)`
  - Output: `blt_cross_attention_weights` view of the cross-attention block.
- `blt_local_byte_grad_view(blt_local_layer_grad* lg)`
  - Output: `blt_transformer_layer_grad` view of the byte-transformer block gradients for `blt_transformer_layer_backward`.
- `blt_local_byte_layer_config(embed_dim, num_heads, rope_theta, hidden_dim, local_mask, rope_cos_view, rope_sin_view)`
  - Output: `blt_transformer_config` for a byte-transformer layer: RMSNorm + SwiGLU + causal RoPE'd self-attention over `local_mask`.

### blt/models/local_encoder.h
- `blt_local_encoder_config` struct
  - Fields:
    - `size_t embed_dim`: Transformer hidden width ($h_E$).
    - `size_t patch_dim`: Width of the patch input ($h_P$); if 0, same as `embed_dim` (no splitting).
    - `size_t num_layers`: Number of byte transformer layers ($l_E$, default: 1).
    - `size_t hidden_dim`: Intermediate hidden width for FFN projections.
    - `size_t num_heads`: Number of byte self-attention heads.
    - `size_t cross_attn_heads`: Number of cross-attention heads ($U_E$).
    - `size_t local_window`: Sliding window size ($w_E$) for byte self-attention (0 = full causal).
    - `bool cross_attn_all_layers`: If `false`, cross-attention fires only after the final layer.
    - `blt_patch_pool_type pool_type`: Initialization strategy for initial patch representations $P_0$ (default: MEAN).
    - `blt_hash_ngram_config ngram_config`: Hash n-gram config; `ngram_config.embed_dim` must match `embed_dim`.
    - `float rope_theta`: Base frequency for Rotary Position Embeddings.
    - `size_t max_seq_len`: Maximum sequence length used to size the shared RoPE cache.
- `blt_local_encoder_layer_storage` struct
  - Alias of `blt_local_layer_storage` (see blt/models/local_common.h): `norm1_weight`, `attn_qkv_w`, `attn_proj_w`, `norm2_weight`, `ffn_up_w`, `ffn_gate_w`, `ffn_down_w`, `cross_norm_weight`, `cross_weight_q`, `cross_weight_k`, `cross_weight_v`, `cross_weight_proj` — owned storage for layer weights.
- `blt_local_encoder` struct
  - Fields:
    - `blt_local_encoder_config config`: Encoder configuration parameters.
    - `blt_tensor byte_embedding_weight`: Byte lookup table `[256, embed_dim]`.
    - `blt_hash_ngram_weights ngram_weights`: Hash n-gram tables with small-uniform initialization.
    - `blt_tensor rope_cos_cache`, `rope_sin_cache`: Precomputed RoPE tables `[max_seq_len, head_dim/2]`.
    - `blt_local_encoder_layer_storage* layers`: Array of per-layer storage blocks `[num_layers]`.
- `blt_local_encoder_layer_grad` struct
  - Alias of `blt_local_layer_grad` (see blt/models/local_common.h) — gradient storage per layer.
- `blt_local_encoder_grad` struct
  - Fields:
    - `blt_tensor embedding_grad`: Gradient table for byte embeddings `[256, embed_dim]` (scatter-add target).
    - `blt_hash_ngram_weights ngram_grads`: Gradient tables for n-grams (scatter-add target).
    - `blt_local_encoder_layer_grad* layer_grads`: Layer gradient structures `[num_layers]` (overwritten).
- `blt_local_encoder_create(arena, config)`
  - Input: `arena` for allocations and `config` parameters.
  - Output: Allocated `blt_local_encoder*` handle with zero-initialized weights and precomputed RoPE caches.
  - Behavior: Allocates model storage, creates small-uniform n-gram tables, and precomputes RoPE cache once for the entire encoder. Caller populates model weights afterward.
- `blt_local_encoder_grad_create(arena, model)`
  - Input: `arena` for storage and reference `model`.
  - Output: Allocated, zero-initialized `blt_local_encoder_grad*` structure matching `model` shapes.
  - Behavior: Prepares zero-initialized gradient structures for tracking backpropagation.
- `blt_local_encoder_forward(model, bytes_in, patches, num_patches, doc_boundaries, num_docs, patch_out, byte_hidden_out, arena)`
  - Input: `model`, raw input bytes `bytes_in` `[seq_len]` (UINT8), `patches` array `[num_patches]` tiling `[0, seq_len)`, optional `doc_boundaries` array and `num_docs`, caller-allocated `patch_out`, `byte_hidden_out`, and scratch `arena`.
  - Output: Writes final patch representations $P_{\text{final}}$ into `patch_out` `[num_patches, embed_dim]` and final byte representations $h_{\text{final}}$ into `byte_hidden_out` `[seq_len, embed_dim]`.
  - Behavior: Embeds bytes + n-grams, constructs $P_0$ via patch pooling, executes $l_E$ transformer layers with sliding-window self-attention, and updates patch representations using block-diagonal cross-attention.
- `blt_local_encoder_backward(model, bytes_in, patches, num_patches, doc_boundaries, num_docs, grad_patch_out, grad_byte_hidden_out, grad, arena)`
  - Input: Same forward parameters, plus `grad_patch_out` `[num_patches, embed_dim]` ($dL/dP_{\text{final}}$), optional `grad_byte_hidden_out` `[seq_len, embed_dim]` ($dL/dh_{\text{final}}$), destination gradient handle `grad`, and scratch `arena`.
  - Output: Populates `grad` struct with parameter gradients.
  - Behavior: Recomputes forward passes with per-layer caching, then propagates gradients in reverse through cross-attention, byte transformer layers, patch pooling, n-gram tables, and byte embeddings.


### blt/models/local_decoder.h
- `blt_local_decoder_config` struct
  - Fields:
    - `size_t embed_dim`: Transformer hidden width (h_D).
    - `size_t patch_dim`: Width of the patch input ($h_P$); if 0, same as `embed_dim` (no splitting).
    - `size_t num_layers`: Number of decoder layers (l_D).
    - `size_t hidden_dim`: Intermediate hidden width for FFN projections.
    - `size_t num_heads`: Number of byte self-attention heads.
    - `size_t cross_attn_heads`: Number of cross-attention heads.
    - `size_t local_window`: Sliding window size (0 = full causal).
    - `bool cross_attn_all_layers`: If `true`, cross-attention fires in every layer (decoder default paper finding; configurable).
    - `float rope_theta`: Base frequency for Rotary Position Embeddings.
    - `size_t max_seq_len`: Maximum sequence length used to size the shared RoPE cache.
    - `size_t vocab_size`: Vocabulary size for the LM head (typically 256).
- `blt_local_decoder_layer_storage` struct
  - Alias of `blt_local_layer_storage` (see blt/models/local_common.h). Cross-attention block weights (`cross_norm_weight`, `cross_weight_q`, `cross_weight_k`, `cross_weight_v`, `cross_weight_proj`) and byte-transformer block weights (`norm1_weight`, `attn_qkv_w`, `attn_proj_w`, `norm2_weight`, `ffn_up_w`, `ffn_gate_w`, `ffn_down_w`).
- `blt_local_decoder` struct
  - Fields:
    - `blt_local_decoder_config config`: Decoder configuration parameters.
    - `blt_tensor rope_cos_cache`, `rope_sin_cache`: Precomputed RoPE tables `[max_seq_len, head_dim/2]`.
    - `blt_local_decoder_layer_storage* layers`: Array of per-layer storage blocks `[num_layers]`.
    - `blt_tensor lm_head_weight`: Language-model head weights `[embed_dim, vocab_size]`.
- `blt_local_decoder_layer_grad` struct
  - Alias of `blt_local_layer_grad` (see blt/models/local_common.h) — per-layer gradient storage mirroring `blt_local_decoder_layer_storage`.
- `blt_local_decoder_grad` struct
  - Fields:
    - `blt_local_decoder_layer_grad* layer_grads`: Layer gradient structures `[num_layers]` (overwritten).
    - `blt_tensor lm_head_grad`: Gradient for LM head weights.
- `blt_local_decoder_create(arena, config)`
  - Input: `arena` for allocations and `config` parameters.
  - Output: Allocated `blt_local_decoder*` handle with zero-initialized weights and precomputed RoPE caches.
  - Behavior: Allocates model storage and precomputes shared RoPE cache once for the entire decoder. Caller populates model weights afterward.
- `blt_local_decoder_grad_create(arena, model)`
  - Input: `arena` for storage and reference `model`.
  - Output: Allocated, zero-initialized `blt_local_decoder_grad*` structure matching `model` shapes.
  - Behavior: Prepares zero-initialized gradient structures for tracking backpropagation.
- `blt_local_decoder_forward(model, byte_hidden_in, patch_in, patches, num_patches, bytes_in, doc_boundaries, num_docs, logits_out, loss_out, arena)`
  - Input: `model`, `byte_hidden_in` `[seq_len, embed_dim]` (encoder byte outputs), `patch_in` `[num_patches, embed_dim]` (global transformer outputs), `patches` array, optional `bytes_in` targets `[seq_len]` (UINT8), `doc_boundaries`, `num_docs`, and scratch `arena`.
  - Output: Writes token logits into `logits_out` `[seq_len, vocab_size]` and optional scalar loss into `loss_out`.
  - Behavior: Runs cross-attention from patch representations into causal byte transformer layers, applies causal local self-attention (sliding window or full causal), and projects final byte hidden states through the LM head to produce logits and loss.
- `blt_local_decoder_backward(model, byte_hidden_in, patch_in, patches, num_patches, bytes_in, doc_boundaries, num_docs, grad_byte_hidden_in, grad_patch_in, grad, arena)`
  - Input: Same forward parameters plus destination gradients `grad_byte_hidden_in` `[seq_len, embed_dim]` (feeds back into local encoder), `grad_patch_in` `[num_patches, embed_dim]` (feeds back into global transformer), destination gradient handle `grad`, and scratch `arena`.
  - Output: Populates `grad` struct with parameter gradients and writes upstream gradients into `grad_byte_hidden_in` and `grad_patch_in` when provided.


### blt/models/model.h
- `blt_model_config` struct
  - Fields:
    - `blt_local_encoder_config encoder_config`: configuration for the byte-level local encoder.
    - `blt_global_transformer_config global_config`: configuration for the patch-level global transformer.
    - `blt_local_decoder_config decoder_config`: configuration for the local decoder head.
- `blt_model` struct
  - Fields:
    - `blt_model_config config`: model configuration snapshot used at creation time.
    - `blt_local_encoder* encoder`: pointer to the local byte encoder submodule.
    - `blt_global_transformer* global`: pointer to the global patch transformer submodule.
    - `blt_local_decoder* decoder`: pointer to the decoder submodule that emits logits and loss.
- `blt_model_grad` struct
  - Fields:
    - `blt_local_encoder_grad* encoder_grad`: gradients for the encoder.
    - `blt_global_transformer_grad* global_grad`: gradients for the global transformer.
    - `blt_local_decoder_grad* decoder_grad`: gradients for the decoder.
- `blt_model_create(arena, config)`
  - Input: arena for storage and model configuration that defines the encoder, global transformer, and decoder widths.
  - Output: allocates and initializes a full model object. The encoder/global/decoder `embed_dim` values must match.
  - Behavior: constructs the local encoder, global transformer, and local decoder into one composed model instance.
- `blt_model_grad_create(arena, model)`
  - Input: arena for storage and a fully initialized model.
  - Output: allocates a gradient container with one gradient struct per submodule.
  - Behavior: creates gradient holders for the encoder, global transformer, and decoder so backward propagation can accumulate updates.
- `blt_model_forward(model, bytes_in, patches, num_patches, doc_boundaries, num_docs, logits_out, loss_out, arena)`
  - Input: model pointer, input byte tensor `[seq_len]` in `UINT8`, patch metadata array, number of patches, document boundary indices, number of documents, output logits tensor, scalar loss tensor, and scratch arena.
  - Output: writes logits `[seq_len, vocab_size]` and scalar loss into `logits_out` and `loss_out`.
  - Behavior: runs the encoder on the raw bytes and patch spans, remaps document boundaries from byte offsets to patch indices, runs the global transformer over patch features, and finally decodes to token logits/loss. `loss_out` is a 0D or scalar tensor carrying the sequence loss.
- `blt_model_backward(model, bytes_in, patches, num_patches, doc_boundaries, num_docs, grad, arena)`
  - Input: model, byte input, patch metadata, doc boundaries, gradient accumulator object, and scratch arena.
  - Output: recomputes the forward intermediates and backpropagates through the encoder, global transformer, and decoder, accumulating gradients into `grad`.
  - Behavior: mirrors the forward pass in reverse, reusing the same byte-to-patch remapping and the submodule backward routines. The gradient object must be allocated with `blt_model_grad_create`.

------------------------------------------------------------------------------------------------------------
## Core Implementations

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

### src/model/entropy_lm.c
- `blt_sgd_step(...)`
  - Plain elementwise `param -= lr * grad`, CPU-only, no optimizer state.
- `blt_entropy_lm_create(...)` / `blt_entropy_lm_grad_create(...)`
  - Allocate the model's/gradient's weight tensors (embedding, per-layer transformer weights, LM head) and precompute the shared RoPE cache once.
- `blt_entropy_lm_forward(...)`
  - Embedding → N `blt_transformer_forward` calls → LM head matmul → shifted-target `blt_cross_entropy_forward`.
- `blt_entropy_lm_backward(...)`
  - Recomputes each layer's forward pass with caching (mirroring `blt_multihead_attention_backward`'s own recompute pattern, since `blt_transformer_forward` doesn't expose intermediates), then walks the graph in reverse using each op's existing backward.