// Isolated single-patch paper-rule redirect diagnostic: for redirect target r
// in 1..M-1, one decoder forward applies the Fast BLT (3.1.1) cross-attention
// rule to patch r only (its non-final bytes move to group r-1, its final byte
// stays on group r) while every other patch keeps the repo rule (all bytes on
// their own group j). Encoder and global run once per window on the real
// patches; only the decoder sees the synthetic groups. Rows before patch r
// keep their group ids and must match the patch_final baseline exactly (hard
// invariant); only patch r's non-final bytes are scored. Ground truth per row
// i is text[i+1].
//
// Usage: eval_isolate_redirect --checkpoint MODEL --corpus FILE --dump OUT
//                              --baseline-tsv PATH [options]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "core/allocator.h"
#include "core/backend.h"
#include "core/tensor.h"
#include "models/model.h"
#include "models/checkpoint.h"
#include "models/model_builder.h"
#include "train_eval.h"
#include "ops/patch_pool.h"

// Patch closure types
enum { CLOSURE_NAT = 0, CLOSURE_MAX, CLOSURE_BND, CLOSURE_COUNT };

// Baseline lookup table bounds: the baseline TSV holds window indices in
// [0, 512) and row indices in [0, 512) for the 512-byte eval window.
#define BASE_MAX_WINDOWS 512
#define BASE_MAX_ROWS 512

// A full 500-window sweep must reproduce the part-1 headline row count: the
// non-final rows of every patch j >= 1 across all windows.
#define EXPECTED_FULL_WINDOWS 500
#define EXPECTED_SCORED_ROWS 205910

static long fsize(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    return sz;
}

// Short closure label for dump lines and reports.
static const char *closure_str(int closure) {
    switch (closure) {
    case CLOSURE_NAT:
        return "nat";
    case CLOSURE_MAX:
        return "max";
    default:
        return "bnd";
    }
}

// Classify a patch's closure: boundary (last patch), max-length capped, or natural.
static int classify_closure(const blt_patch_info *patches, size_t M, size_t k, size_t max_patch_len) {
    if (k == M - 1) return CLOSURE_BND;
    if (patches[k].length >= max_patch_len) return CLOSURE_MAX;
    return CLOSURE_NAT;
}

// Find the patch index containing row i.
static size_t find_patch(const blt_patch_info *patches, size_t M, size_t i) {
    for (size_t k = 0; k < M; k++) {
        if (i >= patches[k].start_idx && i < patches[k].start_idx + patches[k].length) return k;
    }
    return 0;
}

// Build the synthetic run array applying the paper rule to redirect target r
// only: entries below r-1 keep their real patch, entry r-1 covers patch r-1
// whole plus patch r's non-final bytes, entry r shrinks to patch r's final
// byte, entries above r keep their real patch. Requires 1 <= r < M.
static void build_single_fake(const blt_patch_info *patches, size_t M, size_t window, size_t r, blt_patch_info *fake) {
    size_t end_r = patches[r].start_idx + patches[r].length;
    if (end_r > window) {
        fprintf(stderr, "[ISOLATE] construction FAIL: patch %zu ends at %zu > window %zu\n", r, end_r, window);
        exit(1);
    }
    for (size_t g = 0; g < M; g++) {
        size_t start_g = patches[g].start_idx;
        size_t end_g = start_g + patches[g].length;
        size_t run_start;
        size_t run_end;
        if (g < r - 1) {
            run_start = start_g;
            run_end = end_g;
        } else if (g == r - 1) {
            run_start = start_g;
            run_end = end_r - 1;
        } else if (g == r) {
            run_start = end_r - 1;
            run_end = end_r;
        } else {
            run_start = start_g;
            run_end = end_g;
        }
        fake[g].start_idx = run_start;
        fake[g].length = run_end - run_start;
        fake[g].peak_entropy = 0.f;
    }
}

// Return the run index containing byte i in the synthetic array, or (size_t)-1 if absent.
static size_t run_index_of(const blt_patch_info *fake, size_t M, size_t i) {
    for (size_t g = 0; g < M; g++) {
        if (i >= fake[g].start_idx && i < fake[g].start_idx + fake[g].length) return g;
    }
    return (size_t)-1;
}

// Expected run index for byte i when only patch r is redirected: bytes of any
// patch other than r keep their own patch index; patch r's final byte keeps r
// and its non-final bytes move to r-1 (r >= 1 so r-1 >= 0).
static size_t expected_single(const blt_patch_info *patches, size_t M, size_t r, size_t i) {
    size_t j = find_patch(patches, M, i);
    size_t ends = patches[j].start_idx + patches[j].length;
    int is_final = (i == ends - 1);
    if (j != r) return j;
    return is_final ? r : r - 1;
}

// Verify the synthetic array tiles [0, window) with M runs of length >= 1 and
// that every real byte's run index matches the isolated-redirect rule.
// Aborts on failure.
static int verify_single_construction(const blt_patch_info *patches, size_t M, size_t window, size_t r,
                                      const blt_patch_info *fake) {
    if (M < 1 || r < 1 || r >= M) {
        fprintf(stderr, "[ISOLATE] construction FAIL: M=%zu r=%zu out of range\n", M, r);
        return 0;
    }
    size_t pos = 0;
    for (size_t g = 0; g < M; g++) {
        if (fake[g].start_idx != pos) {
            fprintf(stderr, "[ISOLATE] construction FAIL: run %zu starts at %zu, expected %zu\n", g, fake[g].start_idx,
                    pos);
            return 0;
        }
        if (fake[g].length < 1) {
            fprintf(stderr, "[ISOLATE] construction FAIL: run %zu length %zu < 1\n", g, fake[g].length);
            return 0;
        }
        pos += fake[g].length;
    }
    if (pos != window) {
        fprintf(stderr, "[ISOLATE] construction FAIL: runs cover %zu bytes, window=%zu\n", pos, window);
        return 0;
    }
    for (size_t j = 0; j < M; j++) {
        size_t start = patches[j].start_idx;
        size_t ends = start + patches[j].length;
        for (size_t i = start; i < ends; i++) {
            size_t want = expected_single(patches, M, r, i);
            size_t got = run_index_of(fake, M, i);
            if (got != want) {
                fprintf(stderr, "[ISOLATE] construction FAIL: byte %zu (patch %zu) run=%zu expected=%zu (r=%zu)\n", i,
                        j, got, want, r);
                return 0;
            }
        }
    }
    return 1;
}

// Selftest one redirect target on one synthetic patch set: construction
// invariants, blt_patch_build_group_ids equivalence with expected_single,
// rows before r keeping their own patch index, and the length-1 redirect
// target degenerating to the real patches.
static int selftest_case(const char *name, const blt_patch_info *patches, size_t M, size_t window, size_t r) {
    blt_patch_info fake[32];
    size_t id_out[32];
    size_t byte_out[32];
    if (M > 32 || window > 32 || r < 1 || r >= M) {
        fprintf(stderr, "[SELFCHECK] FAIL %s: case exceeds fixed buffers or r=%zu out of [1, %zu)\n", name, r, M);
        return 0;
    }
    build_single_fake(patches, M, window, r, fake);
    if (!verify_single_construction(patches, M, window, r, fake)) return 0;
    blt_patch_build_group_ids(fake, M, window, id_out, byte_out);
    for (size_t g = 0; g < M; g++) {
        if (id_out[g] != g) {
            fprintf(stderr, "[SELFCHECK] FAIL %s: id_out[%zu]=%zu expected=%zu\n", name, g, id_out[g], g);
            return 0;
        }
    }
    for (size_t i = 0; i < window; i++) {
        size_t want = expected_single(patches, M, r, i);
        if (byte_out[i] != want) {
            fprintf(stderr, "[SELFCHECK] FAIL %s: group_ids[%zu]=%zu expected=%zu\n", name, i, byte_out[i], want);
            return 0;
        }
    }
    size_t start_r = patches[r].start_idx;
    for (size_t i = 0; i < start_r; i++) {
        size_t want = find_patch(patches, M, i);
        if (byte_out[i] != want) {
            fprintf(stderr, "[SELFCHECK] FAIL %s: pre-redirect row %zu group=%zu expected patch %zu\n", name, i,
                    byte_out[i], want);
            return 0;
        }
    }
    if (patches[r].length == 1) {
        for (size_t g = 0; g < M; g++) {
            if (fake[g].start_idx != patches[g].start_idx || fake[g].length != patches[g].length) {
                fprintf(stderr, "[SELFCHECK] FAIL %s: len_r=1 but run %zu differs from real patch\n", name, g);
                return 0;
            }
        }
    }
    fprintf(stderr, "[SELFCHECK] %s: PASS\n", name);
    return 1;
}

// Synthetic redirect cases: r=1, interior r, r=M-1, length-1 redirect targets
// (construction degenerates to the real patches), and all-length-1 patches.
static int run_selftest(void) {
    blt_patch_info p_mix[4] = {{0, 5, 0.f}, {5, 4, 0.f}, {9, 7, 0.f}, {16, 6, 0.f}};
    blt_patch_info p_len1mid[4] = {{0, 4, 0.f}, {4, 3, 0.f}, {7, 1, 0.f}, {8, 6, 0.f}};
    blt_patch_info p_all1[4] = {{0, 1, 0.f}, {1, 1, 0.f}, {2, 1, 0.f}, {3, 1, 0.f}};

    int ok = 1;
    ok &= selftest_case("r=1", p_mix, 4, 22, 1);
    ok &= selftest_case("r=interior", p_mix, 4, 22, 2);
    ok &= selftest_case("r=M-1", p_mix, 4, 22, 3);
    ok &= selftest_case("len_r=1 interior", p_len1mid, 4, 14, 2);
    ok &= selftest_case("len_r=1 r=1", p_all1, 4, 4, 1);
    ok &= selftest_case("all-len1 r=M-1", p_all1, 4, 4, 3);
    if (!ok) {
        fprintf(stderr, "[SELFCHECK] FAIL\n");
        return 1;
    }
    fprintf(stderr, "[SELFCHECK] PASS\n");
    return 0;
}

// Baseline correct flags loaded from the patch_final 8-column TSV.
static uint8_t base_present[BASE_MAX_WINDOWS][BASE_MAX_ROWS];
static uint8_t base_value[BASE_MAX_WINDOWS][BASE_MAX_ROWS];

// Load (window, row) -> correct from the baseline TSV, skipping its header
// line. Aborts on malformed or out-of-range rows so a broken baseline cannot
// silently skew the pre-flip invariant.
static int load_baseline(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Error: cannot read baseline TSV '%s'\n", path);
        return 0;
    }
    char line[512];
    size_t lineno = 0;
    size_t loaded = 0;
    while (fgets(line, sizeof(line), f)) {
        lineno++;
        if (lineno == 1) continue;
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;
        int w, row, patch_idx, patch_start, patch_len, is_final, correct;
        char closure[16];
        int fields = sscanf(line, "%d\t%d\t%d\t%d\t%d\t%15[^\t]\t%d\t%d", &w, &row, &patch_idx, &patch_start,
                            &patch_len, closure, &is_final, &correct);
        if (fields != 8 || w < 0 || row < 0 || w >= BASE_MAX_WINDOWS || row >= BASE_MAX_ROWS ||
            (correct != 0 && correct != 1)) {
            fprintf(stderr, "Error: baseline TSV '%s' line %zu malformed\n", path, lineno);
            fclose(f);
            return 0;
        }
        base_present[w][row] = 1;
        base_value[w][row] = (uint8_t)correct;
        loaded++;
    }
    fclose(f);
    if (loaded == 0) {
        fprintf(stderr, "Error: baseline TSV '%s' has no data rows\n", path);
        return 0;
    }
    fprintf(stderr, "[ISOLATE] loaded baseline: %s (%zu rows)\n", path, loaded);
    return 1;
}

// Baseline correct flag for (window, row); aborts when the key is missing.
static int baseline_lookup(size_t w, size_t row) {
    if (w >= BASE_MAX_WINDOWS || row >= BASE_MAX_ROWS || !base_present[w][row]) {
        fprintf(stderr, "Error: baseline TSV missing key (window=%zu row=%zu)\n", w, row);
        exit(1);
    }
    return base_value[w][row];
}

// Top-1 correctness of byte-row i against ground truth text[i+1].
static int row_correct(const float *logits, size_t i, const uint8_t *text) {
    const float *row = logits + i * 256;
    int best = 0;
    for (int v = 1; v < 256; v++)
        if (row[v] > row[best]) best = v;
    return best == (int)text[i + 1];
}

int main(int argc, char **argv) {
    const char *checkpoint = NULL;
    const char *corpus = NULL;
    const char *entropy_lm_path = NULL;
    const char *dump_path = NULL;
    const char *baseline_path = NULL;
    size_t window = 512;
    size_t embed = 256;
    size_t hidden = 512;
    size_t enc_layers = 2;
    size_t glob_layers = 6;
    size_t dec_layers = 2;
    int cross_last = 0;
    int diffusion = 1;
    size_t num_windows = 50;
    size_t skip = 0;
    size_t max_patch_len = 16;
    int selftest = 0;
    int use_cuda = 0;
    // -2 = sweep r in 1..M-1, -1 = control (no redirect), >=1 = single fixed target
    long redirect_patch = -2;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--checkpoint") && i + 1 < argc) checkpoint = argv[++i];
        else if (!strcmp(argv[i], "--corpus") && i + 1 < argc) corpus = argv[++i];
        else if (!strcmp(argv[i], "--window") && i + 1 < argc) window = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--embed") && i + 1 < argc) embed = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--hidden") && i + 1 < argc) hidden = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--enc-layers") && i + 1 < argc) enc_layers = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--glob-layers") && i + 1 < argc) glob_layers = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--dec-layers") && i + 1 < argc) dec_layers = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--cross-attn") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "last")) cross_last = 1;
        } else if (!strcmp(argv[i], "--diffusion") && i + 1 < argc) diffusion = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--num-windows") && i + 1 < argc) num_windows = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--skip") && i + 1 < argc) skip = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--entropy-lm") && i + 1 < argc) entropy_lm_path = argv[++i];
        else if (!strcmp(argv[i], "--max-patch-length") && i + 1 < argc) max_patch_len = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump_path = argv[++i];
        else if (!strcmp(argv[i], "--baseline-tsv") && i + 1 < argc) baseline_path = argv[++i];
        else if (!strcmp(argv[i], "--redirect-patch") && i + 1 < argc) redirect_patch = atol(argv[++i]);
        else if (!strcmp(argv[i], "--selftest")) selftest = 1;
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "cuda")) use_cuda = 1;
            else if (!strcmp(argv[i], "cpu")) use_cuda = 0;
            else {
                fprintf(stderr, "unknown --backend '%s'\n", argv[i]);
                return 1;
            }
        }
    }

    if (selftest) return run_selftest();

    int control = (redirect_patch == -1);
    int fixed_mode = (redirect_patch >= 1);
    if (redirect_patch == 0 || redirect_patch < -2) {
        fprintf(stderr, "invalid --redirect-patch %ld: use -1 (control), omitted (sweep), or >=1 (single target)\n",
                redirect_patch);
        return 1;
    }

    if (!checkpoint || !corpus || !dump_path || (!control && !baseline_path)) {
        fprintf(stderr, "Usage: eval_isolate_redirect --checkpoint MODEL --corpus FILE --dump OUT [options]\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --window W           Window size (default 512)\n");
        fprintf(stderr, "  --num-windows N      Number of windows to evaluate (default 50)\n");
        fprintf(stderr, "  --skip S             Window offset before first window (default 0)\n");
        fprintf(stderr, "  --entropy-lm PATH    Entropy LM for patching\n");
        fprintf(stderr, "  --max-patch-length L Max patch length for classification (default 16)\n");
        fprintf(stderr, "  --dump PATH          Per-row dump output file (required)\n");
        fprintf(stderr, "  --baseline-tsv PATH  patch_final baseline TSV (required unless control mode)\n");
        fprintf(stderr, "  --redirect-patch R   -1 = control (no redirect); omitted = sweep r=1..M-1;\n");
        fprintf(stderr, "                       >=1 = single fixed redirect target\n");
        fprintf(stderr, "  --embed/--hidden/--enc-layers/--glob-layers/--dec-layers  Model shape\n");
        fprintf(stderr, "  --cross-attn all|last  Cross-attention placement\n");
        fprintf(stderr, "  --diffusion 0|1      Diffusion mode (default 1)\n");
        fprintf(stderr, "  --backend cpu|cuda   Compute backend (default cpu)\n");
        fprintf(stderr, "  --selftest           Run synthetic construction self-check and exit\n");
        return 1;
    }

    long file_size = fsize(corpus);
    if (file_size <= 0) {
        fprintf(stderr, "Error: cannot read corpus '%s'\n", corpus);
        return 1;
    }
    size_t data_len = (size_t)file_size;
    uint8_t *data = (uint8_t *)malloc(data_len);
    FILE *f = fopen(corpus, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open corpus '%s'\n", corpus);
        return 1;
    }
    fread(data, 1, data_len, f);
    fclose(f);

    if (!control && !load_baseline(baseline_path)) return 1;

    blt_backend dev = use_cuda ? BLT_BACKEND_CUDA : BLT_BACKEND_CPU;
    blt_arena *arena = blt_arena_create(1024ULL * 1024 * 1024, dev);

    blt_model_config cfg;
    blt_model_config_defaults(&cfg, embed, hidden, enc_layers, glob_layers, dec_layers, window, cross_last);
    blt_model *model = blt_model_create(arena, &cfg);
    blt_model_load(model, checkpoint);
    fprintf(stderr, "[ISOLATE] loaded checkpoint: %s\n", checkpoint);
    fprintf(stderr, "[ISOLATE] config: embed=%zu hidden=%zu enc=%zu glob=%zu dec=%zu cross=%s diff=%d win=%zu\n", embed,
            hidden, enc_layers, glob_layers, dec_layers, cross_last ? "last" : "all", diffusion, window);
    fprintf(stderr, "[ISOLATE] max_patch_length=%zu entropy_lm=%s backend=%s mode=%s\n", max_patch_len,
            entropy_lm_path ? entropy_lm_path : "(none)", use_cuda ? "cuda" : "cpu",
            control ? "control" : (fixed_mode ? "fixed-r" : "sweep"));

    blt_entropy_lm *el = NULL;
    if (entropy_lm_path) {
        el = blt_make_entropy_lm(arena, window, entropy_lm_path, 11);
        fprintf(stderr, "[ISOLATE] loaded entropy LM: %s\n", entropy_lm_path);
    }

    // win_arena holds per-window inputs and encoder/global/logits tensors
    // (reset per window); pass_arena holds per-pass decoder intermediates
    // (reset per redirect pass so passes never accumulate allocations).
    blt_arena *win_arena = blt_arena_create(1024ULL * 1024 * 1024, dev);
    blt_arena *pass_arena = blt_arena_create(256ULL * 1024 * 1024, dev);

    FILE *dump_file = fopen(dump_path, "w");
    if (!dump_file) {
        fprintf(stderr, "Error: cannot open dump file '%s'\n", dump_path);
        return 1;
    }
    if (control) {
        fprintf(dump_file, "window\trow\tpatch_idx\tpatch_start\tpatch_len\tclosure\tis_final\tcorrect\n");
    } else {
        fprintf(dump_file, "window\trow\tredirect_patch_idx\tpatch_start\tpatch_len\tclosure\tpos_in_patch\tcorrect\t"
                           "baseline_correct\n");
    }

    size_t inv_path_len = strlen(dump_path) + 5;
    char *inv_path = (char *)malloc(inv_path_len);
    snprintf(inv_path, inv_path_len, "%s.inv", dump_path);
    FILE *inv_file = fopen(inv_path, "w");
    if (!inv_file) {
        fprintf(stderr, "Error: cannot open invariant dump file '%s'\n", inv_path);
        return 1;
    }

    float *L = (float *)malloc(window * 256 * sizeof(float));
    uint8_t *corr = (uint8_t *)malloc(window);

    size_t total_windows = 0;
    size_t passes = 0;
    size_t scored_rows = 0;
    size_t scored_correct = 0;
    size_t pre_rows_total = 0;
    size_t pre_flips = 0;
    size_t post_rows = 0;
    size_t post_flips = 0;
    size_t skipped_r0 = 0;
    size_t skipped_m128 = 0;

    for (size_t w = 0; w < num_windows; w++) {
        size_t offset = (skip + w) * window;
        if (offset + window + 1 > data_len) break;
        const uint8_t *text = data + offset;
        size_t abs_w = skip + w;

        blt_patch_info patches[128];
        size_t M;
        if (el) {
            M = entropy_segment(win_arena, el, text, window, patches, 128);
            if (M >= 128) {
                skipped_m128++;
                blt_arena_reset(win_arena);
                continue;
            }
        } else {
            M = fixed_stride(window, 4, patches);
        }

        // Encoder and global run once per window on the real patches: their
        // outputs do not depend on the decoder's synthetic redirect groups.
        size_t bshape[1] = {window};
        blt_tensor bytes_in = blt_tensor_create(win_arena, bshape, 1, BLT_DTYPE_UINT8);
        blt_tensor_upload(&bytes_in, text, window);

        size_t p_shape[2] = {M, model->config.encoder_config.embed_dim};
        blt_tensor P = blt_tensor_create(win_arena, p_shape, 2, BLT_DTYPE_FP32);
        size_t h_shape[2] = {window, model->config.encoder_config.embed_dim};
        blt_tensor h = blt_tensor_create(win_arena, h_shape, 2, BLT_DTYPE_FP32);
        blt_local_encoder_forward(model->encoder, &bytes_in, patches, M, NULL, 0, &P, &h, win_arena);

        blt_tensor O = blt_tensor_create(win_arena, p_shape, 2, BLT_DTYPE_FP32);
        blt_global_transformer_forward(model->global, &P, NULL, 0, &O, win_arena);

        size_t lg[2] = {window, 256};
        blt_tensor logits = blt_tensor_create(win_arena, lg, 2, BLT_DTYPE_FP32);
        size_t sc[1] = {1};
        blt_tensor loss = blt_tensor_create(win_arena, sc, 1, BLT_DTYPE_FP32);

        size_t window_passes = 0;

        if (control) {
            // Control: no redirect; the decoder sees the real patches, so the
            // dump is bit-identical to patch_final_split with the same args.
            blt_local_decoder_forward(model->decoder, &h, &O, patches, M, &bytes_in, NULL, NULL, 0, &logits, &loss,
                                      win_arena);
            blt_tensor_download(&logits, L, logits.numel * sizeof(float));
            for (size_t i = 0; i < window; i++) {
                size_t patch_k = find_patch(patches, M, i);
                int closure = classify_closure(patches, M, patch_k, max_patch_len);
                int is_final = (i == patches[patch_k].start_idx + patches[patch_k].length - 1);
                fprintf(dump_file, "%zu\t%zu\t%zu\t%zu\t%zu\t%s\t%d\t%d\n", abs_w, i, patch_k,
                        patches[patch_k].start_idx, patches[patch_k].length, closure_str(closure), is_final,
                        row_correct(L, i, text));
            }
            passes++;
            window_passes = 1;
        } else {
            // r==0 is never a valid redirect target: patch 0 has no previous patch.
            skipped_r0++;
            blt_patch_info fake[128];
            long r_lo = 1;
            long r_hi = (long)M - 1;
            if (fixed_mode) r_lo = r_hi = redirect_patch;
            for (long r = r_lo; r <= r_hi; r++) {
                build_single_fake(patches, M, window, (size_t)r, fake);
                if (!verify_single_construction(patches, M, window, (size_t)r, fake)) exit(1);

                // Decoder intermediates come from pass_arena, reset per pass;
                // h/O/logits/loss live in win_arena and persist for the window.
                blt_local_decoder_forward(model->decoder, &h, &O, fake, M, &bytes_in, NULL, NULL, 0, &logits, &loss,
                                          pass_arena);
                blt_arena_reset(pass_arena);
                blt_tensor_download(&logits, L, logits.numel * sizeof(float));
                for (size_t i = 0; i < window; i++) corr[i] = (uint8_t)row_correct(L, i, text);

                size_t start_r = patches[r].start_idx;
                size_t end_r = start_r + patches[r].length;
                int closure_r = classify_closure(patches, M, (size_t)r, max_patch_len);

                // Invariant: every row before patch r keeps its group id under
                // the isolated redirect, so its correct flag must match the
                // baseline. Violations are printed immediately, not aborted.
                size_t pass_pre_flips = 0;
                for (size_t i = 0; i < start_r; i++) {
                    int base_c = baseline_lookup(abs_w, i);
                    if (corr[i] != base_c) {
                        pass_pre_flips++;
                        pre_flips++;
                        fprintf(stderr, "[INVARIANT] window %zu r %ld pre_flip at row %zu\n", abs_w, r, i);
                    }
                }
                pre_rows_total += start_r;

                // Scored rows: only patch r's non-final bytes.
                for (size_t i = start_r; i < end_r - 1; i++) {
                    int base_c = baseline_lookup(abs_w, i);
                    fprintf(dump_file, "%zu\t%zu\t%ld\t%zu\t%zu\t%s\t%zu\t%d\t%d\n", abs_w, i, r, start_r,
                            patches[r].length, closure_str(closure_r), i - start_r, corr[i], base_c);
                    scored_rows++;
                    scored_correct += (size_t)corr[i];
                }

                // Rows after patch r are informational: causal byte
                // self-attention can mix the redirect into them.
                for (size_t i = end_r; i < window; i++) {
                    int base_c = baseline_lookup(abs_w, i);
                    post_rows++;
                    if (corr[i] != base_c) post_flips++;
                }

                fprintf(inv_file, "%zu\t%ld\t%zu\t%zu\n", abs_w, r, start_r, pass_pre_flips);
                passes++;
                window_passes++;
            }
        }

        total_windows++;
        blt_arena_reset(win_arena);
        blt_arena_reset(pass_arena);
        fprintf(stderr, "[ISOLATE] window %zu M=%zu passes=%zu\n", abs_w, M, window_passes);
    }

    fclose(dump_file);
    fclose(inv_file);

    fprintf(stderr, "\n=== ISOLATED SINGLE-PATCH REDIRECT ===\n");
    fprintf(stderr, "Windows evaluated:   %zu\n", total_windows);
    fprintf(stderr, "Passes:              %zu\n", passes);
    fprintf(stderr, "Scored rows:         %zu (correct %zu)\n", scored_rows, scored_correct);
    fprintf(stderr, "Pre rows checked:    %zu\n", pre_rows_total);
    fprintf(stderr, "Pre flips:           %zu (must be 0)\n", pre_flips);
    fprintf(stderr, "Post rows/flips:     %zu/%zu (informational)\n", post_rows, post_flips);
    fprintf(stderr, "Skipped r==0:        %zu (patch 0 has no previous patch)\n", skipped_r0);
    fprintf(stderr, "Skipped M>=128:      %zu\n", skipped_m128);
    fprintf(stderr, "Wrote %zu dump rows to %s, %zu invariant lines to %s\n",
            control ? total_windows * window : scored_rows, dump_path, control ? 0 : passes, inv_path);

    // Machine-readable output (stdout reserved for this line; shards sum key=int)
    printf("windows=%zu passes=%zu scored_rows=%zu scored_correct=%zu pre_rows=%zu pre_flips=%zu "
           "skipped_r0=%zu skipped_m128=%zu post_rows=%zu post_flips=%zu\n",
           total_windows, passes, scored_rows, scored_correct, pre_rows_total, pre_flips, skipped_r0, skipped_m128,
           post_rows, post_flips);

    int status = 0;
    if (!control && !fixed_mode && total_windows == EXPECTED_FULL_WINDOWS) {
        if (scored_rows == EXPECTED_SCORED_ROWS) {
            fprintf(stderr, "[ISOLATE] scored-row check: PASS (%zu == %d)\n", scored_rows, EXPECTED_SCORED_ROWS);
        } else {
            fprintf(stderr, "[ISOLATE] scored-row check: FAIL (%zu != %d)\n", scored_rows, EXPECTED_SCORED_ROWS);
            status = 1;
        }
    } else if (!control) {
        fprintf(stderr, "[ISOLATE] scored-row check: %zu rows (expectation %d applies to a full %d-window sweep)\n",
                scored_rows, EXPECTED_SCORED_ROWS, EXPECTED_FULL_WINDOWS);
    }

    free(inv_path);
    free(corr);
    free(L);
    free(data);
    return status;
}
