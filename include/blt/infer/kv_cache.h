// blt/infer/kv_cache.h
#ifndef BLT_INFER_KV_CACHE_H
#define BLT_INFER_KV_CACHE_H

#include <stddef.h>

#include "blt/core/tensor.h"
#include "blt/core/allocator.h"
#include "blt/models/local_decoder.h"
#include "blt/models/patcher.h"


// Two-tier KV cache for cache-aware incremental decoding (Fast-BLT Phase C).
//
// Tier 1 (byte window): per-layer self-attention K/V for decoder byte
//   positions. Row index == absolute byte position; rollback = truncate.
// Tier 2 (patch sub-tokens): per-layer cross-attention K/V over the
//   expanded patch sub-tokens ([num_patches*k, E] split form). Append-only:
//   thanks to the prefix-stable entropy patcher, completed patches are
//   immutable, so only the trailing in-progress patch ever needs a rewrite
//   after a verification round commits new bytes.
//
// Validity invariant: self-attn K/V rows [0, self_len) and cross-attn K/V
// sub-tokens of patches [0, cross_num_patches) were computed under exactly
// the same prefix bytes / patch latents that the current segmentation
// implies, so reusing them is bit-identical to recomputing (causality +
// deterministic segmentation). Rows beyond self_len hold scratch data from
// rolled-back drafts and are overwritten before reuse (write-through).
//
// The decode step is unified: new K/V rows are written into the caches
// FIRST, then attention runs over cache[0 .. base+n) -- so prefill chunks
// (n > 1) and single-row draft steps (n == 1) share one code path.
//
// This module composes existing public ops (matmul, rmsnorm, swiglu, the
// masked-softmax inline) rather than adding backend kernels; a future CUDA
// port would map these compositions 1:1 onto fused kernels.
//
// Scope note (Phase C v1): encoder + global transformer still run densely
// once per round (tier 3, patch-level global KV, is deferred until
// measured), and the dense decoder path remains untouched for training /
// parity.

typedef struct {
    const blt_local_decoder* decoder;
    size_t max_seq_len;
    size_t max_subtokens;          // capacity of tier 2 (= max_seq_len; patches have >= 1 byte)
    size_t embed_dim;
    size_t k;                      // patch_dim / embed_dim split factor (1 = no split)
    size_t patch_dim;

    // tier 1: [num_layers] tensors, each [max_seq_len, embed_dim]
    blt_tensor* self_k;
    blt_tensor* self_v;
    size_t self_len;               // valid byte positions [0, self_len)

    // tier 2: [num_layers] tensors, each [max_subtokens, embed_dim]
    blt_tensor* cross_k;
    blt_tensor* cross_v;
    size_t cross_num_patches;      // patches whose sub-token K/V are valid

    blt_patch_info* cached_patches;   // boundary snapshot of cached patches [max_seq_len]
} blt_kv_cache;


// Allocates zeroed K/V buffers for every layer plus bookkeeping storage.
blt_kv_cache* blt_kv_cache_create(blt_arena* arena, const blt_local_decoder* decoder,
                                  size_t max_seq_len);

// Invalidates everything (lengths -> 0). Buffer contents stay as-is.
void blt_kv_cache_reset(blt_kv_cache* cache);

// Rolls the cache back to self-attn validity [0, self_len) and cross-attn
// validity for the first num_patches patches. Both may only shrink relative
// to what was previously marked valid OR grow to cover entries this caller
// has just refreshed; no data is cleared.
void blt_kv_cache_truncate(blt_kv_cache* cache, size_t self_len, size_t num_patches);

// Longest common prefix between the cached patch boundaries and `patches`.
// Returns how many leading patches are identical (start_idx AND length);
// 0 when nothing is cached yet. This is the number of patches whose tier-2
// entries (and whose bytes' tier-1 entries) provably remain valid.
size_t blt_kv_cache_common_patches(const blt_kv_cache* cache,
                                   const blt_patch_info* patches, size_t num_patches);

// Recomputes tier-2 cross-attn K/V for patches [from_patch, num_patches)
// against patch_in [num_patches, patch_dim] (split form internally).
// Only firing layers (cross_attn_placement) write entries, mirroring the
// dense path. Callers must first truncate() to from_patch.
void blt_kv_cache_refresh_cross(blt_kv_cache* cache,
                                const blt_tensor* patch_in,
                                const blt_patch_info* patches, size_t num_patches,
                                size_t from_patch,
                                blt_arena* arena);

// Cache-aware incremental decoder forward for n new byte rows.
//
// d0_rows  [n, embed_dim]: prepared decoder inputs D_0 for absolute byte
//          positions [self_len, self_len+n) (h_final copies for prefill
//          chunks, policy-filled rows for draft/MASK positions).
// logits_out [n, vocab_size]: LM-head projections of the new rows' final
//          states (row r corresponds to absolute position self_len+r).
//
// Updates both tiers: appends n rows to tier 1 and consumes tier-2 entries
// which must already cover all patches touched by the new rows (call
// refresh_cross first). Bit-identical to running the dense
// blt_local_decoder_forward_ext over the same prefix + these rows.
void blt_kv_decode_step(blt_kv_cache* cache,
                        const blt_tensor* patch_in,
                        const blt_patch_info* patches, size_t num_patches,
                        const blt_tensor* d0_rows,
                        blt_tensor* logits_out,
                        blt_arena* arena);

#endif // BLT_INFER_KV_CACHE_H
