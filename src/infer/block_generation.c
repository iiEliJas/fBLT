#include "blt/infer/block_generation.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "blt/core/backend.h"
#include "blt/infer/self_speculation.h"

// splitmix64 stream shared by top-p sampling (deterministic per seed).
static uint64_t rng_next(uint64_t* state) {
    *state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = *state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

//----------------------------------------------------------------------
// Selection kernels (section 3.1.2)
//----------------------------------------------------------------------

uint32_t blt_unmask_select_confidence(const float* scores,
                                      const uint8_t* masked, size_t B,
                                      float alpha) {
    uint32_t sel = 0;
    size_t best = SIZE_MAX;
    for (size_t b = 0; b < B; b++) {
        if (!masked[b]) continue;
        if (best == SIZE_MAX || scores[b] > scores[best]) best = b;
        if (scores[b] >= alpha) sel |= (uint32_t)1u << b;
    }
    if (sel == 0 && best != SIZE_MAX) sel = (uint32_t)1u << best; // progress rule
    return sel;
}

uint32_t blt_unmask_select_eb(const float* scores,
                              const uint8_t* masked, size_t B,
                              float gamma) {
    // Ascending-entropy order via insertion sort over masked indices.
    size_t order[32];
    size_t n = 0;
    for (size_t b = 0; b < B; b++) {
        if (!masked[b]) continue;
        size_t j = n++;
        while (j > 0 && scores[order[j - 1]] > scores[b]) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = b;
    }
    if (n == 0) return 0;

    // Largest prefix with cumulative entropy <= gamma; fallback to the
    // single lowest-entropy cell when even that exceeds the budget.
    uint32_t sel = 0;
    float cum = 0.0f;
    size_t taken = 0;
    while (taken < n &&
           (taken == 0 || cum + scores[order[taken]] <= gamma)) {
        cum += scores[order[taken]];
        sel |= (uint32_t)1u << order[taken];
        taken++;
    }
    if (sel == 0) sel = (uint32_t)1u << order[0];
    return sel;
}

//----------------------------------------------------------------------
// Row statistics + prediction sampling over one logits row
//----------------------------------------------------------------------

typedef struct {
    float p_max;
    float entropy;   // nats
} row_stats;

static void row_analyze(const float* row, size_t V, float temperature_scale_unused,
                        row_stats* out) {
    (void)temperature_scale_unused;
    float mx = row[0];
    for (size_t v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
    double sum = 0.0;
    for (size_t v = 0; v < V; v++) sum += exp((double)row[v] - (double)mx);
    double pmax = 0.0, H = 0.0;
    for (size_t v = 0; v < V; v++) {
        const double p = exp((double)row[v] - (double)mx) / sum;
        if (p > pmax) pmax = p;
        if (p > 1e-12) H -= p * log(p);
    }
    out->p_max = (float)pmax;
    out->entropy = (float)H;
}

// Greedy argmax over a logits row.
static uint8_t row_argmax(const float* row, size_t V) {
    size_t best = 0;
    float bv = row[0];
    for (size_t v = 1; v < V; v++) {
        if (row[v] > bv) { bv = row[v]; best = v; }
    }
    return (uint8_t)best;
}

// Sample from the top-p nucleus of a softmaxed logits row.
static uint8_t row_sample_top_p(const float* row, size_t V, float top_p,
                                uint64_t* rng) {
    // Sort indices by descending probability (insertion sort, V <= 512).
    size_t idx[512];
    double p[512];
    float mx = row[0];
    for (size_t v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
    double sum = 0.0;
    for (size_t v = 0; v < V; v++) { p[v] = exp((double)row[v] - (double)mx); sum += p[v]; }
    for (size_t v = 0; v < V; v++) { p[v] /= sum; idx[v] = v; }
    for (size_t i = 1; i < V; i++) {
        const size_t ki = idx[i];
        const double pi = p[ki];
        size_t j = i;
        while (j > 0 && p[idx[j - 1]] < pi) { idx[j] = idx[j - 1]; j--; }
        idx[j] = ki;
    }

    double cum = 0.0;
    size_t nucleus = V - 1;
    for (size_t i = 0; i < V; i++) {
        cum += p[idx[i]];
        if (cum >= (double)top_p) { nucleus = i; break; }
    }
    double r = (double)(rng_next(rng) >> 11) * (1.0 / 9007199254740992.0); // [0,1)
    double acc = 0.0;
    for (size_t i = 0; i <= nucleus; i++) {
        acc += p[idx[i]];
        if ((float)r < (float)acc) return (uint8_t)idx[i];
    }
    return (uint8_t)idx[nucleus];
}

//----------------------------------------------------------------------
// Adaptive-B rule (rolling acceptance -> next block size)
//----------------------------------------------------------------------

size_t blt_block_adapt_b(size_t cur_b, double rolling_acceptance,
                         size_t b_min, size_t b_max, float target) {
    if (b_max == 0 || b_max < b_min) return cur_b;
    size_t lo = b_min ? b_min : 1;
    if (lo > b_max) lo = b_max;
    size_t b = cur_b < lo ? lo : cur_b;
    b = b > b_max ? b_max : b;
    if (!(rolling_acceptance >= 0.0)) return b;   // NaN: no history yet
    if (rolling_acceptance > (double)(target + BLT_ADAPT_B_MARGIN)) {
        if (b < b_max) b++;
    } else if (rolling_acceptance < (double)(target - BLT_ADAPT_B_MARGIN)) {
        if (b > lo) b--;
    }
    return b;
}

//----------------------------------------------------------------------
// blt_draft_block: Algorithm 1 inner loop
//----------------------------------------------------------------------

#define BLT_BLOCKGEN_MAX_B 32

size_t blt_draft_block(
    const blt_model* model,
    const blt_model_enc_out* enc,
    const blt_patch_info* patches, size_t num_patches,
    const uint8_t* prefix, size_t prefix_len,
    const blt_block_gen_config* config,
    uint8_t* out_block,
    blt_arena* scratch
) {
    BLT_REQUIRE(model != NULL && enc != NULL && patches != NULL && prefix != NULL &&
                config != NULL && out_block != NULL && scratch != NULL,
        "blt_draft_block: arguments cannot be NULL");
    const size_t B = config->block_size;
    BLT_REQUIRE(B >= 1 && B <= BLT_BLOCKGEN_MAX_B,
        "blt_draft_block: block_size must be in [1, 32]");
    BLT_REQUIRE(prefix_len + B <= model->config.decoder_config.max_seq_len,
        "blt_draft_block: prefix_len + block_size exceeds decoder max_seq_len");

    const blt_local_decoder* dec = model->decoder;
    const size_t V = model->config.decoder_config.vocab_size;

    // Inference batch: one live block after the clean prefix. All block rows
    // cross-attend the last latent o_M (paper section 3.1.1); t = 0 keeps
    // every loss path inert.
    blt_block_batch batch;
    memset(&batch, 0, sizeof(batch));
    batch.num_clean = prefix_len;
    batch.block_size = B;
    batch.num_blocks = 1;
    batch.n_block_rows = B;
    batch.t = 0.0f;
    batch.loss_scale = 0.0f;

    uint32_t tokens[BLT_BLOCKGEN_MAX_B];
    uint8_t masked[BLT_BLOCKGEN_MAX_B];
    for (size_t b = 0; b < B; b++) {
        tokens[b] = BLT_MASK_TOKEN_ID;
        masked[b] = 1;
    }

    // NOTE: scratch is NOT reset here -- enc->byte_hidden_out/global_out
    // typically live in this arena and must survive every pass. Callers
    // bound memory with their own round-level arena markers.
    uint64_t rng = config->opts.seed ? config->opts.seed : 1;
    size_t nfes = 0;
    size_t remaining_passes = B; // hard bound: at least one cell per pass

    while (remaining_passes-- > 0) {
        // Refresh batch views from the current block state.
        uint32_t tok32[BLT_BLOCKGEN_MAX_B];
        size_t pos[BLT_BLOCKGEN_MAX_B];
        uint8_t targets[BLT_BLOCKGEN_MAX_B];
        uint8_t valid[BLT_BLOCKGEN_MAX_B];
        uint8_t cell_masked[BLT_BLOCKGEN_MAX_B];
        size_t groups[BLT_BLOCKGEN_MAX_B];
        for (size_t b = 0; b < B; b++) {
            tok32[b] = tokens[b];
            pos[b] = prefix_len + b;
            targets[b] = 0;
            valid[b] = 1;
            cell_masked[b] = masked[b];
            groups[b] = num_patches - 1;   // all rows attend o_M
        }
        batch.tokens = tok32;
        batch.positions = pos;
        batch.targets = targets;
        batch.cell_valid = valid;
        batch.cell_masked = cell_masked;
        batch.groups = groups;

        size_t logits_shape[2] = {prefix_len + B, V};
        blt_tensor logits = blt_tensor_create(scratch, logits_shape, 2, BLT_DTYPE_FP32);

        blt_local_decoder_forward_diffusion_infer(
            dec, &enc->byte_hidden_out, &enc->global_out,
            patches, num_patches, &batch, config->d0_mode, &logits, scratch);
        nfes++;

        // Per-cell statistics and predictions from this pass. Row analysis
        // and sampling run on a host copy of the logits.
        float* rows_host = (float*)malloc(logits.numel * sizeof(float));
        BLT_REQUIRE(rows_host != NULL, "blockgen: logits staging alloc failed");
        blt_tensor_download(&logits, rows_host, logits.numel * sizeof(float));
        float conf[BLT_BLOCKGEN_MAX_B], ent[BLT_BLOCKGEN_MAX_B];
        uint8_t pred[BLT_BLOCKGEN_MAX_B];
        const float* rows = rows_host;
        for (size_t b = 0; b < B; b++) {
            const float* row = rows + (prefix_len + b) * V;
            row_stats st;
            row_analyze(row, V, 1.0f, &st);
            conf[b] = st.p_max;
            ent[b] = st.entropy;
            pred[b] = config->opts.use_top_p
                ? row_sample_top_p(row, V, config->opts.top_p, &rng)
                : row_argmax(row, V);
        }

        const uint32_t sel = (config->opts.strategy == BLT_UNMASK_CONFIDENCE)
            ? blt_unmask_select_confidence(conf, masked, B, config->opts.threshold)
            : blt_unmask_select_eb(ent, masked, B, config->opts.threshold);
        free(rows_host);

        int any = 0;
        for (size_t b = 0; b < B; b++) {
            if (sel & ((uint32_t)1u << b)) {
                tokens[b] = pred[b];
                masked[b] = 0;
                any = 1;
            }
        }
        (void)any;

        int done = 1;
        for (size_t b = 0; b < B; b++) done &= !masked[b];
        if (done) break;
    }

    for (size_t b = 0; b < B; b++) {
        BLT_REQUIRE(tokens[b] < 256, "blt_draft_block: non-byte token survived");
        out_block[b] = (uint8_t)tokens[b];
    }
    return nfes;
}

//----------------------------------------------------------------------
// Outer loops (Algorithm 1)
//----------------------------------------------------------------------

static void segment_prefix(blt_arena* arena, const blt_entropy_lm* entropy_model,
                           const blt_patcher_config* patcher_config,
                           const uint8_t* bytes, size_t len,
                           blt_patch_info* patches, size_t max_patches,
                           size_t* num_patches_out) {
    // Mirrors compute_entropy_vals in self_speculation.c bit-for-bit so
    // drafting and verification always agree on patch boundaries.
    size_t shape1[1] = {len};
    blt_tensor bytes_in = blt_tensor_create(arena, shape1, 1, BLT_DTYPE_UINT8);
    blt_tensor_upload(&bytes_in, bytes, len);

    size_t logits_shape[2] = {len, 256};
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    size_t scalar_shape[1] = {1};
    blt_tensor discard_loss = blt_tensor_create(arena, scalar_shape, 1, BLT_DTYPE_FP32);

    blt_entropy_lm_forward(entropy_model, &bytes_in, &logits, &discard_loss, arena);

    size_t probs_shape[2] = {len, 256};
    blt_tensor probs = blt_tensor_create(arena, probs_shape, 2, BLT_DTYPE_FP32);
    blt_softmax(&logits, &probs);

    size_t vals_shape[1] = {len};
    blt_tensor ent_tensor = blt_tensor_create(arena, vals_shape, 1, BLT_DTYPE_FP32);
    blt_entropy_config entropy_cfg = {.vocab_size = 256, .use_log2 = false};
    blt_compute_entropy(&probs, &ent_tensor, &entropy_cfg);

    // The patcher is host-only: stage device entropies through host memory.
    if (ent_tensor.backend == BLT_BACKEND_CPU) {
        *num_patches_out = blt_segment_patches(&ent_tensor, bytes, patches,
                                               max_patches, patcher_config);
        return;
    }
    float* ent_host = (float*)malloc(len * sizeof(float));
    BLT_REQUIRE(ent_host != NULL, "segment_for_generation: staging alloc failed");
    blt_tensor_download(&ent_tensor, ent_host, len * sizeof(float));
    blt_tensor ent_view;
    view_1d(&ent_view, ent_host, len, BLT_DTYPE_FP32, BLT_BACKEND_CPU);
    *num_patches_out = blt_segment_patches(&ent_view, bytes, patches,
                                           max_patches, patcher_config);
    free(ent_host);
}

static void generate_common(
    const blt_model* model,
    const blt_entropy_lm* entropy_model,
    const blt_patcher_config* patcher_config,
    const uint8_t* prompt_bytes, size_t prompt_len,
    size_t max_new_bytes,
    uint8_t* output_bytes,
    const blt_block_gen_config* config,
    int do_verify,
    blt_infer_stats* stats,
    blt_arena* scratch
) {
    BLT_REQUIRE(model != NULL && entropy_model != NULL && patcher_config != NULL &&
                prompt_bytes != NULL && output_bytes != NULL && config != NULL &&
                scratch != NULL,
        "blockdiff generation: arguments cannot be NULL");
    BLT_REQUIRE(prompt_len >= 1, "blockdiff generation: prompt must be non-empty");
    const size_t target_len = prompt_len + max_new_bytes;
    BLT_REQUIRE(target_len <= model->config.decoder_config.max_seq_len,
        "blockdiff generation: total length exceeds decoder max_seq_len");
    // The verification model may differ from the drafting model, but both
    // must cover the same sequence budget.
    const blt_model* verifier = config->verifier_model ? config->verifier_model : model;
    BLT_REQUIRE(target_len <= verifier->config.decoder_config.max_seq_len,
        "blockdiff generation: target length exceeds verifier max_seq_len");

    memcpy(output_bytes, prompt_bytes, prompt_len);
    size_t l = prompt_len;

    // Adaptive-B rolling state.
    const int adaptive = (do_verify && config->B_max > 0 &&
                          config->adapt_window > 0);
    size_t cur_B = config->block_size;
    double acc_hist[64];
    size_t hist_len = 0, hist_pos = 0;
    if (config->adapt_window > 64) {
        BLT_FATAL("blockdiff generation: adapt_window must be <= 64");
    }

    // Per-round view of the config carrying the current block size into
    // the Algorithm 1 inner loop.
    blt_block_gen_config rcfg = *config;

    while (l < target_len) {
        const size_t round_marker = scratch->offset;

        size_t B = cur_B;
        if (l + B > target_len) B = target_len - l;
        rcfg.block_size = B;

        // Segment + encode the committed prefix once (Algorithm 1 lines 3-4).
        enum { MAX_PATCHES = 256 };
        blt_patch_info patches[MAX_PATCHES];
        size_t num_patches = 0;
        segment_prefix(scratch, entropy_model, patcher_config,
                       output_bytes, l, patches, MAX_PATCHES, &num_patches);
        BLT_REQUIRE(num_patches >= 1, "blockdiff generation: empty segmentation");
        BLT_REQUIRE(patcher_config->max_patch_length * num_patches >= l ||
                    num_patches < MAX_PATCHES,
                    "blockdiff generation: patch array exhausted; increase "
                    "max_patch_length or the segment budget");

        size_t shape1[1] = {l};
        blt_tensor prefix_bytes = blt_tensor_create(scratch, shape1, 1, BLT_DTYPE_UINT8);
        blt_tensor_upload(&prefix_bytes, output_bytes, l);

        blt_model_enc_out enc;
        blt_model_encode(model, &prefix_bytes, patches, num_patches, NULL, 0, &enc, scratch);
        if (stats) stats->nfe_encoder_global++;

        uint8_t draft[BLT_BLOCKGEN_MAX_B];
        const size_t passes = blt_draft_block(model, &enc, patches, num_patches,
                                              output_bytes, l, &rcfg, draft, scratch);
        if (stats) stats->nfe_decoder += passes;
        if (stats) stats->bytes_drafted += B;

        if (!do_verify) {
            memcpy(output_bytes + l, draft, B);
            if (stats) stats->bytes_accepted += B;
            l += B;
        } else {
            memcpy(output_bytes + l, draft, B);
            const size_t committed =
                config->boundary_aligned
                ? blt_verify_draft_aligned(verifier, entropy_model,
                                           patcher_config, output_bytes, l, B,
                                           target_len, stats, scratch)
                : blt_verify_draft(verifier, entropy_model, patcher_config,
                                   output_bytes, l, B, target_len, stats, scratch);
            BLT_REQUIRE(committed > l, "blockdiff DV: verify made no progress");

            if (adaptive) {
                // Fraction of this round's drafts that survived commitment.
                const double acc = (double)(committed - l) / (double)B;
                acc_hist[hist_pos] = acc > 1.0 ? 1.0 : (acc < 0.0 ? 0.0 : acc);
                hist_pos = (hist_pos + 1) % config->adapt_window;
                if (hist_len < config->adapt_window) hist_len++;
                double sum = 0.0;
                for (size_t i = 0; i < hist_len; i++) sum += acc_hist[i];
                cur_B = blt_block_adapt_b(cur_B, sum / (double)hist_len,
                                          config->B_min, config->B_max,
                                          config->accept_target);
            }
            l = committed;
        }

        scratch->offset = round_marker;
    }
}

void blt_generate_greedy_blockdiff(
    const blt_model* model,
    const blt_entropy_lm* entropy_model,
    const blt_patcher_config* patcher_config,
    const uint8_t* prompt_bytes, size_t prompt_len,
    size_t max_new_bytes,
    uint8_t* output_bytes,
    const blt_block_gen_config* config,
    blt_infer_stats* stats,
    blt_arena* scratch
) {
    generate_common(model, entropy_model, patcher_config,
                    prompt_bytes, prompt_len, max_new_bytes, output_bytes,
                    config, 0, stats, scratch);
}

void blt_generate_greedy_blockdiff_verify(
    const blt_model* model,
    const blt_entropy_lm* entropy_model,
    const blt_patcher_config* patcher_config,
    const uint8_t* prompt_bytes, size_t prompt_len,
    size_t max_new_bytes,
    uint8_t* output_bytes,
    const blt_block_gen_config* config,
    blt_infer_stats* stats,
    blt_arena* scratch
) {
    generate_common(model, entropy_model, patcher_config,
                    prompt_bytes, prompt_len, max_new_bytes, output_bytes,
                    config, 1, stats, scratch);
}
