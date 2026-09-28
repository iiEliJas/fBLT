// Masked-cell top-1 accuracy A/B for BLT-D block rows: runs the diffusion
// decoder twice per window on an identical batch, changing only the
// cross-attention latent group ids (training o_{i-1} rule vs inference's
// o_{M-1}-for-all rule), so a train/inference group mismatch shows up as an
// accuracy collapse.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/allocator.h"
#include "core/backend.h"
#include "core/tensor.h"
#include "models/model.h"
#include "models/model_builder.h"
#include "models/checkpoint.h"
#include "models/entropy_lm.h"
#include "models/entropy.h"
#include "models/patcher.h"
#include "models/local_encoder.h"
#include "models/local_decoder.h"
#include "models/block_diffusion.h"
#include "ops/softmax.h"
#include "train_eval.h"

#define MASKACC_MAX_T 32
#define MASKACC_V 256
#define MASKACC_MAX_PATCHES 128

// train       : block at a patch start s_i, cross-attention o_{i-1} (training rule)
// train_lastlat: same blocks, cross-attention = last closed latent (convention A/B)
// aligned     : one block after a prefix that ends exactly at a patch start,
//               cross-attention = last closed latent (no clean twin)
// midpatch    : one block after a prefix that ends 1..max_patch_len-1 bytes into an
//               open patch, cross-attention = last closed latent (lagging case)
#define MASKACC_MAX_PATCH_LEN 16
#define MASKACC_MAX_SEQ 4096
#define MASKACC_NCONV 4
static const char *const conv_names[MASKACC_NCONV] = {"train", "train_lastlat", "aligned", "midpatch"};

typedef struct {
    size_t masked_total;
    size_t masked_hits;
    size_t first_total; // cell 0 of each block
    size_t first_hits;
    size_t last_total; // cell B-1 of each block
    size_t last_hits;
    size_t clean_total;
    size_t clean_hits;
    double pmax_sum;
    double ent_sum;
} maskacc_stats;

static int patcher_fixed_global = 0;

static uint64_t rng_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static long fsize(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    return sz;
}

// First-max-wins argmax, strict > from v=1, matching core/generate_greedy.c.
// Normal-approximation 95% CI for a difference of two independent proportions.
// Used for the convention A/B delta, where the two arms share windows but not
// logits, so the samples are independent.
static double diff_ci95(double p1, size_t n1, double p2, size_t n2, double *lo, double *hi) {
    if (n1 == 0 || n2 == 0) {
        *lo = 0.0;
        *hi = 0.0;
        return 0.0;
    }
    const double var = p1 * (1.0 - p1) / (double)n1 + p2 * (1.0 - p2) / (double)n2;
    const double se = sqrt(var);
    const double d = p2 - p1;
    *lo = d - 1.96 * se;
    *hi = d + 1.96 * se;
    return se;
}

static size_t row_argmax(const float *row, size_t vocab) {
    size_t best = 0;
    for (size_t v = 1; v < vocab; v++) {
        if (row[v] > row[best]) best = v;
    }
    return best;
}

// Masked-cell accuracy + mean max-softmax + mean entropy (nats) over the
// masked rows of one decoder pass: rows N+r for r in [0, n_block_rows).
// Also splits accuracy by cell index within the block, so a per-cell gradient
// (cell 0 sits at the patch start and is the only cell a latent update can
// inform) is visible rather than averaged away.
static void accumulate_masked(maskacc_stats *st, const float *L, const blt_block_batch *batch, size_t block_size) {
    for (size_t r = 0; r < batch->n_block_rows; r++) {
        if (!batch->cell_masked[r]) continue;
        const float *row = L + (batch->num_clean + r) * MASKACC_V;
        st->masked_total++;
        const bool hit = row_argmax(row, MASKACC_V) == (size_t)batch->targets[r];
        if (hit) st->masked_hits++;

        const size_t cell = (block_size > 0) ? (r % block_size) : 0;
        if (cell == 0) {
            st->first_total++;
            if (hit) st->first_hits++;
        }
        if (block_size > 0 && cell == block_size - 1) {
            st->last_total++;
            if (hit) st->last_hits++;
        }

        float mx = row[0];
        for (size_t v = 1; v < MASKACC_V; v++)
            if (row[v] > mx) mx = row[v];
        double sum = 0.0;
        for (size_t v = 0; v < MASKACC_V; v++) sum += exp((double)row[v] - (double)mx);
        st->pmax_sum += 1.0 / sum;
        double hent = 0.0;
        for (size_t v = 0; v < MASKACC_V; v++) {
            double p = exp((double)row[v] - (double)mx) / sum;
            if (p > 0.0) hent -= p * log(p);
        }
        st->ent_sum += hent;
    }
}

// Entropy-LM segmentation mirroring training: entropy LM forward over the
// window -> softmax -> blt_compute_entropy (nats), then entropy patching
// with the training patcher config. Same sequence as compute_entropy_vals
// in infer/self_speculation.c plus entropy_segment in train_eval.c.
// When ent_out is non-NULL it receives a malloc'd host array of the per-byte
// entropy, which the novel layouts need for blt_next_starts_patch.
static size_t entropy_segment_local(blt_arena *arena, blt_entropy_lm *lm, const uint8_t *bytes, size_t len,
                                    blt_patch_info *out, size_t max_patches, float **ent_out) {
    size_t bytes_shape[1] = {len};
    blt_tensor bytes_in = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
    blt_tensor_upload(&bytes_in, bytes, len);

    size_t logits_shape[2] = {len, MASKACC_V};
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    size_t scalar_shape[1] = {1};
    blt_tensor discard_loss = blt_tensor_create(arena, scalar_shape, 1, BLT_DTYPE_FP32);
    blt_entropy_lm_forward(lm, &bytes_in, &logits, &discard_loss, arena);

    size_t probs_shape[2] = {len, MASKACC_V};
    blt_tensor probs = blt_tensor_create(arena, probs_shape, 2, BLT_DTYPE_FP32);
    blt_softmax(&logits, &probs);

    size_t vals_shape[1] = {len};
    blt_tensor vals = blt_tensor_create(arena, vals_shape, 1, BLT_DTYPE_FP32);
    blt_entropy_config entropy_cfg = {.vocab_size = MASKACC_V, .use_log2 = false};
    blt_compute_entropy(&probs, &vals, &entropy_cfg);

    blt_patcher_config pcfg;
    blt_make_patcher_cfg(&pcfg, 0, 2.5f, 1.0f, MASKACC_MAX_PATCH_LEN);

    float *vals_host = (float *)malloc(len * sizeof(float));
    BLT_REQUIRE(vals_host != NULL, "entropy_segment_local: staging alloc failed");
    blt_tensor_download(&vals, vals_host, len * sizeof(float));
    blt_tensor vals_view;
    view_1d(&vals_view, vals_host, len, BLT_DTYPE_FP32, BLT_BACKEND_CPU);
    const size_t n = blt_segment_patches(&vals_view, bytes, out, max_patches, &pcfg);
    if (ent_out) {
        *ent_out = vals_host;
    } else {
        free(vals_host);
    }
    return n;
}

// Encoder + global transformer over a clean prefix: bytes [N] -> encoder
// h_final h [N, E] and global latents O [M, E], both allocated from arena
// (the caller's scratch, reset per pass).
static void run_prefix_encode(blt_model *model, blt_arena *arena, const uint8_t *bytes, size_t N,
                              const blt_patch_info *patches, size_t M, blt_tensor *h_out, blt_tensor *O_out) {
    size_t bshape[1] = {N};
    blt_tensor bytes_in = blt_tensor_create(arena, bshape, 1, BLT_DTYPE_UINT8);
    blt_tensor_upload(&bytes_in, bytes, N);

    size_t p_shape[2] = {M, model->config.encoder_config.embed_dim};
    blt_tensor P = blt_tensor_create(arena, p_shape, 2, BLT_DTYPE_FP32);
    size_t h_shape[2] = {N, model->config.encoder_config.embed_dim};
    blt_tensor h = blt_tensor_create(arena, h_shape, 2, BLT_DTYPE_FP32);
    blt_local_encoder_forward(model->encoder, &bytes_in, patches, M, NULL, 0, &P, &h, arena);

    blt_tensor O = blt_tensor_create(arena, p_shape, 2, BLT_DTYPE_FP32);
    blt_global_transformer_forward(model->global, &P, NULL, 0, &O, arena);
    *h_out = h;
    *O_out = O;
}

// Builds one novel layout: prefix = bytes[0, P), then a single block of B
// cells at absolute positions P..P+B-1 with no clean row there, so the block
// has no clean twin to copy from. The prefix is re-segmented independently,
// so the patch count reflects only what a decoder would see at inference.
// Cross-attention targets the last CLOSED latent: if a patch starts exactly at
// P the final prefix patch is closed and we use it, otherwise the trailing
// patch is still open and the last closed latent is one patch stale.
static size_t novel_layout_eval(blt_model *model, blt_arena *scratch, blt_arena *seg_arena, blt_entropy_lm *el,
                                const uint8_t *text, size_t P, size_t block_size, float t, uint64_t seed,
                                blt_d0_mode d0_mode, maskacc_stats *st, int want_per_cell) {
    if (P < 2 || P + block_size > MASKACC_MAX_SEQ) return 0;

    blt_patch_info pp[MASKACC_MAX_PATCHES];
    float *ent = NULL;
    blt_arena_reset(seg_arena);
    const size_t Mp = patcher_fixed_global
                          ? fixed_stride(P, 4, pp)
                          : entropy_segment_local(seg_arena, el, text, P, pp, MASKACC_MAX_PATCHES, &ent);
    if (Mp < 2 || Mp >= MASKACC_MAX_PATCHES) {
        free(ent);
        return 0;
    }

    // Decide whether the final prefix patch is closed, using only x[0..P).
    const blt_patcher_config pcfg = {0};
    blt_patcher_config pc;
    blt_make_patcher_cfg(&pc, 0, 2.5f, 1.0f, MASKACC_MAX_PATCH_LEN);
    (void)pcfg;
    const size_t last_start = pp[Mp - 1].start_idx;
    const size_t last_len = pp[Mp - 1].length;
    int last_closed = 0;
    if (!patcher_fixed_global) {
        last_closed = blt_next_starts_patch(text, P, ent, last_start, last_len, &pc);
    } else {
        last_closed = 1; // fixed stride: the next stride boundary always follows
    }
    const size_t group = last_closed ? (Mp - 1) : (Mp - 2);
    free(ent);

    blt_arena_reset(scratch);
    blt_tensor h, O;
    run_prefix_encode(model, scratch, text, P, pp, Mp, &h, &O);

    blt_block_batch batch;
    memset(&batch, 0, sizeof(batch));
    batch.num_clean = P;
    batch.block_size = block_size;
    batch.num_blocks = 1;
    batch.n_block_rows = block_size;
    batch.t = (t > 0.0f) ? t : 1e-6f;
    batch.loss_scale = 0.0f;
    batch.last_row_scale = 1.0f;
    batch.tokens = (uint32_t *)blt_container_alloc(scratch, block_size * sizeof(uint32_t));
    batch.positions = (size_t *)blt_container_alloc(scratch, block_size * sizeof(size_t));
    batch.targets = (uint8_t *)blt_container_alloc(scratch, block_size);
    batch.cell_valid = (uint8_t *)blt_container_alloc(scratch, block_size);
    batch.cell_masked = (uint8_t *)blt_container_alloc(scratch, block_size);
    batch.groups = (size_t *)blt_container_alloc(scratch, block_size * sizeof(size_t));

    uint64_t rng = seed * 0x9E3779B97F4A7C15ULL + 0x2545F4914F6CDD1DULL;
    for (size_t r = 0; r < block_size; r++) {
        const uint8_t real = text[P + r];
        batch.positions[r] = P + r;
        batch.targets[r] = real;
        batch.cell_valid[r] = 1;
        // Independent per-cell masking at rate t (all masked when t >= 1).
        const bool m = (t >= 1.0f) || ((double)(rng_next(&rng) >> 40) / (double)(1u << 24)) < (double)t;
        batch.cell_masked[r] = m ? 1 : 0;
        batch.tokens[r] = m ? BLT_MASK_TOKEN_ID : (uint32_t)real;
        batch.groups[r] = group;
    }

    size_t S = P + block_size;
    size_t lg[2] = {S, MASKACC_V};
    blt_tensor logits = blt_tensor_create(scratch, lg, 2, BLT_DTYPE_FP32);
    blt_local_decoder_forward_diffusion_infer(model->decoder, &h, &O, pp, Mp, &batch, d0_mode, &logits, scratch);

    float *L = (float *)malloc(logits.numel * sizeof(float));
    BLT_REQUIRE(L != NULL, "novel_layout_eval: logits staging alloc failed");
    blt_tensor_download(&logits, L, logits.numel * sizeof(float));
    accumulate_masked(st, L, &batch, block_size);
    (void)want_per_cell;
    (void)seed;
    free(L);
    return block_size;
}

int main(int argc, char **argv) {
    const char *checkpoint = NULL;
    const char *corpus = NULL;
    const char *entropy_lm_path = NULL;
    size_t embed = 0;
    size_t hidden = 0;
    size_t enc_layers = 0;
    size_t glob_layers = 0;
    size_t dec_layers = 0;
    int cross_last = 0;
    size_t window = 512;
    size_t num_windows = 32;
    size_t block_size = 4;
    float ts[MASKACC_MAX_T];
    size_t n_t = 0;
    float t_min = 0.1f;
    long seed = 7;
    blt_d0_mode d0_mode = BLT_D0_LEARNED;
    int backend_cuda = 0;
    int layout_novel = 0;
    int patcher_fixed = 0;
    int guard = 0;
    int guard_ok = 1;
    size_t sites = 1;
    size_t novel_prefix_max = 256;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--checkpoint") && i + 1 < argc) checkpoint = argv[++i];
        else if (!strcmp(argv[i], "--entropy-lm") && i + 1 < argc) entropy_lm_path = argv[++i];
        else if (!strcmp(argv[i], "--corpus") && i + 1 < argc) corpus = argv[++i];
        else if (!strcmp(argv[i], "--embed") && i + 1 < argc) embed = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--hidden") && i + 1 < argc) hidden = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--enc-layers") && i + 1 < argc) enc_layers = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--glob-layers") && i + 1 < argc) glob_layers = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--dec-layers") && i + 1 < argc) dec_layers = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--cross-attn") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "last")) cross_last = 1;
            else if (strcmp(argv[i], "all") != 0) {
                fprintf(stderr, "Error: --cross-attn must be all|last\n");
                return 1;
            }
        } else if (!strcmp(argv[i], "--window") && i + 1 < argc) window = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--windows") && i + 1 < argc) num_windows = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--block-size") && i + 1 < argc) block_size = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--t") && i + 1 < argc) {
            if (n_t >= MASKACC_MAX_T) {
                fprintf(stderr, "Error: too many --t values (max %d)\n", MASKACC_MAX_T);
                return 1;
            }
            ts[n_t++] = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--t-min") && i + 1 < argc) t_min = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atol(argv[++i]);
        else if (!strcmp(argv[i], "--d0") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "zeros")) d0_mode = BLT_D0_ZEROS;
            else if (strcmp(argv[i], "learned") != 0) {
                fprintf(stderr, "Error: --d0 must be learned|zeros\n");
                return 1;
            }
        } else if (!strcmp(argv[i], "--backend") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "cuda")) backend_cuda = 1;
            else if (strcmp(argv[i], "cpu") != 0) {
                fprintf(stderr, "Error: --backend must be cpu|cuda\n");
                return 1;
            }
        } else if (!strcmp(argv[i], "--layout") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "novel")) layout_novel = 1;
            else if (!strcmp(argv[i], "twin")) layout_novel = 0;
            else {
                fprintf(stderr, "Error: --layout must be twin|novel\n");
                return 1;
            }
        } else if (!strcmp(argv[i], "--sites") && i + 1 < argc) sites = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--novel-prefix-max") && i + 1 < argc) novel_prefix_max = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--guard")) {
            guard = 1;
        } else if (!strcmp(argv[i], "--patcher") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "fixed")) {
                patcher_fixed = 1;
                patcher_fixed_global = 1;
            } else if (strcmp(argv[i], "entropy") != 0) {
                fprintf(stderr, "Error: --patcher must be entropy|fixed\n");
                return 1;
            }
        }
    }

    if (backend_cuda) {
        fprintf(stderr, "FATAL: cuda not supported in this tool (cpu only)\n");
        return 1;
    }
    if (!checkpoint || !corpus || (!patcher_fixed && !entropy_lm_path)) {
        fprintf(stderr, "Usage: masked_acc --checkpoint MODEL --entropy-lm FILE --corpus FILE "
                        "--embed N --hidden N --enc-layers N --glob-layers N --dec-layers N [options]\n"
                        "  --layout twin|novel   evaluation layout (default twin = training layout)\n"
                        "  --t F                 mask probability (repeatable); --guard needs --t 1.0\n"
                        "  --guard               assert |train - novel| < 0.10 and train < 0.99 at t=1.0;\n"
                        "                        exit 1 on failure (CI regression guard)\n"
                        "  --patcher entropy|fixed  segmentation (default entropy; fixed needs no --entropy-lm)\n");
        return 1;
    }
    if (!embed || !hidden || !enc_layers || !glob_layers || !dec_layers) {
        fprintf(stderr, "Error: --embed --hidden --enc-layers --glob-layers --dec-layers are required\n");
        return 1;
    }
    if (window < 8) {
        fprintf(stderr, "Error: --window must be >= 8\n");
        return 1;
    }
    if (block_size < 1) {
        fprintf(stderr, "Error: --block-size must be >= 1\n");
        return 1;
    }
    if (guard && !layout_novel) {
        fprintf(stderr, "Error: --guard requires --layout novel (the guard compares train vs novel)\n");
        return 1;
    }
    if (n_t == 0) {
        ts[0] = 0.5f;
        ts[1] = 1.0f;
        n_t = 2;
    }

    long file_size = fsize(corpus);
    if (file_size <= 0) {
        fprintf(stderr, "Error: cannot read corpus '%s'\n", corpus);
        return 1;
    }
    size_t data_len = (size_t)file_size;
    uint8_t *data = (uint8_t *)malloc(data_len);
    BLT_REQUIRE(data != NULL, "corpus staging alloc failed");
    FILE *f = fopen(corpus, "rb");
    BLT_REQUIRE(f != NULL, "cannot open corpus");
    if (fread(data, 1, data_len, f) != data_len) {
        fprintf(stderr, "Error: failed to read corpus '%s'\n", corpus);
        fclose(f);
        free(data);
        return 1;
    }
    fclose(f);

    blt_backend dev = BLT_BACKEND_CPU;

    // Same max_seq_len headroom as train_blt_d.c: S = window + block_size * (M-1)
    // must fit. Fixed patching has deterministic M = ceil(window/4) up front;
    // entropy patching has no minimum patch length, so its exact M is only known
    // per window and is checked after segmentation below.
    const size_t MS = 1024;
    const size_t M_fixed = (window + 3) / 4;
    const size_t S_worst = patcher_fixed ? window + block_size * (M_fixed - 1) : window;
    BLT_REQUIRE(S_worst <= MS, "--window/--block-size exceed max_seq_len headroom of %zu (S=%zu)", MS, S_worst);

    // Fixed-stride patching writes ceil(window/4) entries with no internal cap (the
    // entropy patcher stops at max_patches), so windows that would overrun the
    // patches[MASKACC_MAX_PATCHES] buffer are rejected before any work starts.
    if (patcher_fixed && M_fixed > MASKACC_MAX_PATCHES) {
        fprintf(stderr, "Error: --window %zu with --patcher fixed yields %zu patches > %d cap\n", window, M_fixed,
                MASKACC_MAX_PATCHES);
        return 1;
    }

    blt_arena *model_arena = blt_arena_create(160ULL * 1024 * 1024, dev);
    blt_arena *scratch = blt_arena_create(256ULL * 1024 * 1024, dev);
    blt_arena *lm_arena = blt_arena_create(32ULL * 1024 * 1024, dev);
    blt_arena *seg_arena = blt_arena_create(16ULL * 1024 * 1024, dev);

    blt_model_config cfg;
    blt_model_config_defaults(&cfg, embed, hidden, enc_layers, glob_layers, dec_layers, MS, cross_last);
    blt_model *model = blt_model_create(model_arena, &cfg);
    blt_model_load(model, checkpoint);
    fprintf(stderr, "[MASKED_ACC] loaded checkpoint: %s\n", checkpoint);
    fprintf(stderr,
            "[MASKED_ACC] config: embed=%zu hidden=%zu enc=%zu glob=%zu dec=%zu cross=%s win=%zu B=%zu "
            "windows=%zu t_min=%.3f seed=%ld d0=%s patcher=%s\n",
            embed, hidden, enc_layers, glob_layers, dec_layers, cross_last ? "last" : "all", window, block_size,
            num_windows, (double)t_min, seed, d0_mode == BLT_D0_LEARNED ? "learned" : "zeros",
            patcher_fixed ? "fixed" : "entropy");

    blt_entropy_lm *el = patcher_fixed ? NULL : blt_make_entropy_lm(lm_arena, MS, entropy_lm_path, 11);
    if (el) fprintf(stderr, "[MASKED_ACC] loaded entropy LM: %s\n", entropy_lm_path);

    maskacc_stats stats[MASKACC_MAX_T][3];
    memset(stats, 0, sizeof(stats));
    size_t evaluated = 0;
    size_t skipped = 0;

    for (size_t w = 0; w < num_windows; w++) {
        size_t offset = w * window;
        if (offset + window > data_len) break;
        const uint8_t *text = data + offset;
        size_t win_clean_total = 0;
        size_t win_clean_hits = 0;

        blt_patch_info patches[MASKACC_MAX_PATCHES];
        blt_arena_reset(seg_arena);
        // Mirrors the training-time selection at csrc/train_blt_d.c:354-356.
        size_t M = patcher_fixed
                       ? fixed_stride(window, 4, patches)
                       : entropy_segment_local(seg_arena, el, text, window, patches, MASKACC_MAX_PATCHES, NULL);
        if (M < 2 || M >= MASKACC_MAX_PATCHES) {
            skipped++;
            continue;
        }
        const int patches_usable = 1;
        // Entropy M has no pre-flight bound, so the exact S = window + block_size *
        // (M-1) is enforced here, before the encoder and decoder see it.
        BLT_REQUIRE(window + block_size * (M - 1) <= MS,
                    "window %zu: S = window + block_size * (M-1) exceeds max_seq_len headroom of %zu (M=%zu)", window,
                    MS, M);

        for (size_t ti = 0; ti < n_t; ti++) {
            for (int ci = 0; ci < 2; ci++) {
                blt_arena_reset(scratch);

                blt_tensor h, O;
                run_prefix_encode(model, scratch, text, window, patches, M, &h, &O);

                blt_block_batch batch;
                blt_block_batch_build_t(&batch, scratch, text, window, patches, M, block_size,
                                        (uint64_t)seed + (uint64_t)w, ts[ti]);
                if (batch.t < t_min) batch.t = t_min;
                batch.loss_scale = 0.0f;
                if (ci == 1) {
                    for (size_t r = 0; r < batch.n_block_rows; r++) batch.groups[r] = M - 1;
                }

                size_t S = window + batch.n_block_rows;
                size_t lg[2] = {S, MASKACC_V};
                blt_tensor logits = blt_tensor_create(scratch, lg, 2, BLT_DTYPE_FP32);
                size_t sc[1] = {1};
                blt_tensor loss = blt_tensor_create(scratch, sc, 1, BLT_DTYPE_FP32);
                blt_local_decoder_forward_diffusion(model->decoder, &h, &O, patches, M, text, NULL, &batch, d0_mode,
                                                    &logits, &loss, scratch);

                float *L = (float *)malloc(logits.numel * sizeof(float));
                BLT_REQUIRE(L != NULL, "logits staging alloc failed");
                blt_tensor_download(&logits, L, logits.numel * sizeof(float));

                maskacc_stats *st = &stats[ti][ci];
                accumulate_masked(st, L, &batch, block_size);

                const size_t clean_t0 = st->clean_total;
                const size_t clean_h0 = st->clean_hits;
                for (size_t i = 0; i + 1 < window; i++) {
                    const float *row = L + i * MASKACC_V;
                    st->clean_total++;
                    if (row_argmax(row, MASKACC_V) == (size_t)text[i + 1]) st->clean_hits++;
                }
                if (ti == 0 && ci == 0) {
                    win_clean_total = st->clean_total - clean_t0;
                    win_clean_hits = st->clean_hits - clean_h0;
                }

                free(L);
            }
        }

        // Novel layouts. Each re-segments its own prefix and places one block
        // where no clean row exists, so there is no clean twin to copy from.
        //   aligned  : prefix ends exactly at a patch start (all prefix patches closed)
        //   midpatch : prefix ends 1..max_patch_len-1 bytes into an open patch, so the
        //              last closed latent lags by one patch (the lagging case)
        // Each site costs a full prefix encode plus a decode, so the prefix is
        // capped at novel_prefix_max bytes and --sites controls how many sites
        // per window are sampled (all sampled sites land in the same stats).
        if (offset + window + block_size <= data_len && patches_usable) {
            for (size_t ti = 0; ti < n_t; ti++) {
                size_t taken = 0;
                for (size_t j = 1; j + 1 < M && taken < sites; j++) {
                    const size_t P = patches[j].start_idx;
                    if (P < 8 || P > novel_prefix_max || P + block_size > window) continue;
                    if (!novel_layout_eval(model, scratch, seg_arena, el, text, P, block_size, ts[ti],
                                           (uint64_t)seed + (uint64_t)w, d0_mode, &stats[ti][2], 1))
                        continue;
                    taken++;
                }
                taken = 0;
                for (size_t j = 1; j + 1 < M && taken < sites; j++) {
                    const size_t s = patches[j].start_idx;
                    const size_t len = patches[j].length;
                    if (len < 2) continue;
                    // Vary the depth into the open patch with w and j so
                    // successive sites probe different lagging offsets.
                    const size_t P = s + 1 + ((size_t)(w + j) % (len - 1));
                    if (P < 8 || P > novel_prefix_max || P + block_size > window) continue;
                    if (!novel_layout_eval(model, scratch, seg_arena, el, text, P, block_size, ts[ti],
                                           (uint64_t)seed + (uint64_t)w, d0_mode, &stats[ti][3], 1))
                        continue;
                    taken++;
                }
            }
        }

        evaluated++;
        fprintf(stderr, "[MASKED_ACC] window %zu/%zu M=%zu done\n", w + 1, num_windows, M);
    }

    fprintf(stderr, "[MASKED_ACC] windows evaluated=%zu skipped=%zu\n", evaluated, skipped);

    const int n_conv = layout_novel ? MASKACC_NCONV : 2;
    printf("t      layout          masked_acc  cell0_acc  cellB-1_acc  masked_n   AR_top1   masked_H\n");
    for (size_t ti = 0; ti < n_t; ti++) {
        for (int ci = 0; ci < n_conv; ci++) {
            const maskacc_stats *st = &stats[ti][ci];
            double macc = st->masked_total ? (double)st->masked_hits / (double)st->masked_total : 0.0;
            double f0 = st->first_total ? (double)st->first_hits / (double)st->first_total : 0.0;
            double fl = st->last_total ? (double)st->last_hits / (double)st->last_total : 0.0;
            double cacc = st->clean_total ? (double)st->clean_hits / (double)st->clean_total : 0.0;
            double hent = st->masked_total ? st->ent_sum / (double)st->masked_total : 0.0;
            printf("%-6.2f %-14s %-11.4f %-10.4f %-12.4f %-10zu %-9.4f %.2f\n", (double)ts[ti], conv_names[ci], macc,
                   f0, fl, st->masked_total, cacc, hent);
        }
    }
    printf("\n");
    for (size_t ti = 0; ti < n_t; ti++) {
        const maskacc_stats *tr = &stats[ti][0];
        const maskacc_stats *tl = &stats[ti][1];
        const maskacc_stats *al = &stats[ti][2];
        const maskacc_stats *mp = &stats[ti][3];
        double ta = tr->masked_total ? (double)tr->masked_hits / (double)tr->masked_total : 0.0;
        double la = tl->masked_total ? (double)tl->masked_hits / (double)tl->masked_total : 0.0;
        double aa = al->masked_total ? (double)al->masked_hits / (double)al->masked_total : 0.0;
        double ma = mp->masked_total ? (double)mp->masked_hits / (double)mp->masked_total : 0.0;
        // Convention A/B: o_{i-1} (training) vs last closed latent, same blocks.
        double lo = 0.0, hi = 0.0;
        diff_ci95(ta, tr->masked_total, la, tl->masked_total, &lo, &hi);
        printf("t=%.2f  convention delta (train_lastlat - train) = %+.4f  95%% CI [%+.4f, %+.4f]  "
               "(o_{i-1} %.4f n=%zu  lastlat %.4f n=%zu)\n",
               (double)ts[ti], la - ta, lo, hi, ta, tr->masked_total, la, tl->masked_total);
        // Aligned vs midpatch: how much does the lagging latent cost?
        double alo = 0.0, ahi = 0.0;
        diff_ci95(aa, al->masked_total, ma, mp->masked_total, &alo, &ahi);
        printf("t=%.2f  aligned %.4f  midpatch %.4f  delta %+.4f  95%% CI [%+.4f, %+.4f]\n", (double)ts[ti], aa, ma,
               ma - aa, alo, ahi);
        // The leak tripwire: train layout must not be far above aligned-novel.
        printf("t=%.2f  train %.4f  aligned-novel %.4f  gap %.4f\n", (double)ts[ti], ta, aa, aa - ta);
    }

    if (layout_novel) {
        // Headline: the aligned novel layout has no clean twin, so it is the
        // number that reflects real draft quality. Report the t=1.0 worst case.
        size_t worst = 0;
        double worst_a = 0.0;
        int have = 0;
        for (size_t ti = 0; ti < n_t; ti++) {
            const maskacc_stats *al = &stats[ti][2];
            if (!al->masked_total) continue;
            const double aa = (double)al->masked_hits / (double)al->masked_total;
            if (!have || aa < worst_a) {
                worst = ti;
                worst_a = aa;
                have = 1;
            }
        }
        if (have) {
            const maskacc_stats *al = &stats[worst][2];
            const maskacc_stats *mp = &stats[worst][3];
            const maskacc_stats *tr = &stats[worst][0];
            const double aa = worst_a;
            const double ma = mp->masked_total ? (double)mp->masked_hits / (double)mp->masked_total : 0.0;
            const double ta = tr->masked_total ? (double)tr->masked_hits / (double)tr->masked_total : 0.0;
            const double f0 = al->first_total ? (double)al->first_hits / (double)al->first_total : 0.0;
            printf("\nHEADLINE  aligned-novel masked_acc @ t=%.2f = %.4f  (cell0 %.4f, midpatch %.4f, train %.4f)\n",
                   (double)ts[worst], aa, f0, ma, ta);
            if (ta - aa >= 0.10) {
                printf("VERDICT: train/novel gap %.4f >= 0.10 -- mask leak present\n", ta - aa);
            } else {
                printf("VERDICT: train/novel gap %.4f < 0.10 -- no mask leak\n", ta - aa);
            }
        }

        if (guard) {
            size_t gi = n_t;
            for (size_t ti = 0; ti < n_t; ti++) {
                if (ts[ti] == 1.0f) {
                    gi = ti;
                    break;
                }
            }
            if (gi >= n_t) {
                fprintf(stderr, "GUARD FAIL: --guard needs a --t 1.0 sweep point\n");
                guard_ok = 0;
            } else {
                const maskacc_stats *gtr = &stats[gi][0];
                const maskacc_stats *gal = &stats[gi][2];
                if (!gtr->masked_total || !gal->masked_total) {
                    fprintf(stderr, "GUARD FAIL: no masked cells at t=1.0 (train %zu, aligned %zu)\n",
                            gtr->masked_total, gal->masked_total);
                    guard_ok = 0;
                } else {
                    const double ga = (double)gtr->masked_hits / (double)gtr->masked_total;
                    const double na = (double)gal->masked_hits / (double)gal->masked_total;
                    const double gap = fabs(ga - na);
                    printf("GUARD t=1.0  train %.4f  aligned-novel %.4f  gap %.4f\n", ga, na, gap);
                    if (!(gap < 0.10)) {
                        fprintf(stderr, "GUARD FAIL: |train - aligned| = %.4f >= 0.10 (mask leak)\n", gap);
                        guard_ok = 0;
                    }
                    if (!(ga < 0.99)) {
                        fprintf(stderr, "GUARD FAIL: train-layout masked acc %.4f >= 0.99 (clean twin leak)\n", ga);
                        guard_ok = 0;
                    }
                }
            }
            printf("GUARD: %s\n", guard_ok ? "PASS" : "FAIL");
        }
    }

    free(data);
    blt_arena_destroy(seg_arena);
    blt_arena_destroy(lm_arena);
    blt_arena_destroy(scratch);
    blt_arena_destroy(model_arena);
    return guard_ok ? 0 : 1;
}
