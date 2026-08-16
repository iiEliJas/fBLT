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
- `blt_tensor_view_2d(blt_tensor* t, void* data, size_t rows, size_t cols, blt_backend backend)`
  - Input: tensor pointer, data pointer, row count, column count, and backend.
  - Output: initializes `t` as a 2D view over the provided data.
- `blt_tensor_view_3d(blt_tensor* t, void* data, size_t d0, size_t d1, size_t d2, blt_backend backend)`
  - Input: tensor pointer, data pointer, three dimension sizes, and backend.
  - Output: initializes `t` as a 3D view over the provided data.


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
  - Behaviour: Validates an N-dimensional FP32 tensor. Pass 0 for any dimension in `dims` to skip that dimension's check.
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
  - Output: a zero-initialized tensor with allocated storage.



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
  - Input: 1D entropy tensor, uint8_t bytes array, output patch buffer, maximum patch count, and patcher config
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
- `blt_transformer_forward(input, weights, output, config, arena)`
  - Input: input sequence tensor, transformer weights, output tensor, transformer config, and scratch arena.
  - Output: writes a single transformer block forward pass result into `output`.
  - Behavior: performs pre-norm self-attention with residual, followed by pre-norm FFN with residual. The `arena` is used for intermediate tensors and is not reset by the function.

### blt/model/entropy_lm.h
Assembly of already-implemented pieces (byte embedding, RoPE-enabled transformer stack, cross-entropy) into one callable "tiny causal byte LM": embedding → N transformer layers → LM head → shifted next-byte cross-entropy loss.
- `blt_entropy_lm_config` struct
  - Fields:
    - `size_t embed_dim`: transformer hidden width.
    - `size_t num_layers`: number of stacked transformer blocks.
    - `size_t hidden_dim`: FFN intermediate width.
    - `size_t num_heads`: attention heads; `embed_dim` must be divisible by `num_heads`.
    - `size_t max_seq_len`: upper bound used to size the shared RoPE cache.
    - `float rope_theta`: RoPE base (e.g. 500000.0f).
- `blt_transformer_layer_storage` struct
  - Fields: `blt_tensor norm1_weight`, `attn_qkv_w`, `attn_proj_w`, `norm2_weight`, `ffn_up_w`, `ffn_gate_w`, `ffn_down_w` — owned storage for one transformer layer's weights.
- `blt_entropy_lm` struct
  - Fields:
    - `blt_entropy_lm_config config`
    - `blt_tensor embedding_weight`: `[256, embed_dim]`.
    - `blt_tensor lm_head_weight`: `[embed_dim, 256]`.
    - `blt_tensor rope_cos_cache`, `rope_sin_cache`: `[max_seq_len, head_dim/2]`, precomputed once at model-init time.
    - `blt_transformer_config layer_config`: single shared config for every layer (`norm_type = BLT_NORM_RMSNORM`, `activation_type = BLT_ACTIVATION_SWIGLU`, `use_rope = true`).
    - `blt_transformer_layer_storage* layer_storage`: owned weight storage, `[num_layers]`.
    - `blt_transformer_weights* layer_weights`: const-pointer views into `layer_storage`, `[num_layers]` — what's actually passed to `blt_transformer_forward`.
- `blt_transformer_layer_grad` struct
  - Fields: `blt_tensor norm1_weight`, `attn_qkv_w`, `attn_proj_w`, `norm2_weight`, `ffn_up_w`, `ffn_gate_w`, `ffn_down_w` — gradient counterpart of `blt_transformer_layer_storage`.
- `blt_entropy_lm_grad` struct
  - Fields:
    - `blt_tensor embedding_grad`: `[256, embed_dim]`.
    - `blt_transformer_layer_grad* layer_grads`: `[num_layers]`.
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
  - Behavior: Validates that `n` is in `[1, 8]` and `prime * modulus` does not overflow `uint64`.
- `blt_rolling_hash_update(state, new_byte)`
  - Input: pointer to `blt_rolling_hash_state` and a `uint8_t` byte.
  - Output: Returns the hash of the current n-gram if `positions_seen >= n`. Returns `UINT64_MAX` if not enough bytes have been seen yet.
  - Behavior: Uses a circular buffer to track the last `n` bytes and updates the hash in O(1) time using the rolling polynomial method: `H_new = (H_old * prime + b_new - b_outgoing * prime^n) mod modulus`.
- `blt_hash_ngram_create(arena, config)`
  - Input: arena for storage and `blt_hash_ngram_config`.
  - Output: Returns an allocated `blt_hash_ngram_weights` struct.
  - Behavior: Allocates embedding tables and initializes them with small uniform random values in `[-0.02, 0.02]` to break symmetry.
- `blt_hash_ngram_forward(weights, config, bytes_in, byte_emb, out, arena)`
  - Input: `weights`, `config`, `bytes_in` (1D UINT8 `[seq_len]`), `byte_emb` (2D FP32 `[seq_len, embed_dim]`), and `arena`.
  - Output: `out` (2D FP32 `[seq_len, embed_dim]`). Contains `byte_emb` + n-gram embeddings.
  - Behavior: For each position `i`, adds the hash table lookups for all active n-gram sizes to the base byte embedding. Positions `i < n-1` receive no contribution from the size-`n` table. If `normalize` is true, scales the final output by `1 / (num_ngram_sizes + 1)`.
- `blt_hash_ngram_backward(weights, config, bytes_in, grad_out, grad_byte_emb, grad_tables, arena)`
  - Input: `weights` (unused but kept for API symmetry), `config`, `bytes_in`, `grad_out` (2D FP32 `[seq_len, embed_dim]`), and `arena`.
  - Output: `grad_byte_emb` (2D FP32 `[seq_len, embed_dim]`) and `grad_tables` (array of 2D FP32 `[per_ngram_vocab, embed_dim]`).
  - Behavior: Computes gradients w.r.t the base byte embeddings and the hash tables. Scales gradients by `1 / (num_ngram_sizes + 1)` if `normalize` is true. Scatter-adds gradients into `grad_tables` (which MUST be zero-initialized by the caller). To avoid aliasing issues if `grad_byte_emb` and `grad_out` point to the same memory, the scatter-add is executed strictly before overwriting `grad_byte_emb`.




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