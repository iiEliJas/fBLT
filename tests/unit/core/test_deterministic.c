#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Platform compatibility for access() and temp paths
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#define blt_access _access
#define BLT_F_OK 0
#define BLT_R_OK 4
#define BLT_X_OK 0
#define BLT_DEVNULL "NUL"
static const char *blt_temp_dir(void) {
    static char buf[260];
    GetTempPathA(sizeof(buf), buf);
    return buf;
}
#else
#include <unistd.h>
#define blt_access access
#define BLT_F_OK F_OK
#define BLT_R_OK R_OK
#define BLT_X_OK X_OK
#define BLT_DEVNULL "/dev/null"
static const char *blt_temp_dir(void) { return "/tmp"; }
#endif

// Build a temp file path: <temp_dir>/<filename>
static void blt_temp_path(char *buf, size_t len, const char *filename) {
    snprintf(buf, len, "%s%s%s", blt_temp_dir(),
#ifdef _WIN32
             "\\",
#else
             "/",
#endif
             filename);
}

#include "test_helpers.h"

// Regression test: run the training binary twice with --deterministic and
// assert the gradnorm logs are byte-identical. This catches any CUDA
// nondeterminism reintroduced by future changes.

int run_deterministic_regression_test(void) {
    // Skip if the training binary or corpus is not available
    if (blt_access("./bin/train_blt_d", BLT_X_OK) != 0) {
        fprintf(stderr, "  [SKIP] bin/train_blt_d not built; skipping deterministic test\n");
        return 1;
    }
    if (blt_access("data/train.bin", BLT_R_OK) != 0) {
        fprintf(stderr, "  [SKIP] data/train.bin not found; skipping deterministic test\n");
        return 1;
    }

    // Build temp file paths
    char log1[512], log2[512], loss1[512], loss2[512];
    blt_temp_path(log1, sizeof(log1), "blt_det_test1.log");
    blt_temp_path(log2, sizeof(log2), "blt_det_test2.log");
    blt_temp_path(loss1, sizeof(loss1), "blt_det_loss1.log");
    blt_temp_path(loss2, sizeof(loss2), "blt_det_loss2.log");

    // Use a small config: 200 steps, seed=7, deterministic
#ifdef _WIN32
    const char *bin_path = "./bin/train_blt_d.exe";
#else
    const char *bin_path = "./bin/train_blt_d";
#endif

    char cmd1[2048], cmd2[2048];
    snprintf(cmd1, sizeof(cmd1),
             "%s --corpus data/train.bin --steps 200 --lr 0.05 "
             "--block-size 4 --window 48 --embed 64 --hidden 128 --layers 2 "
             "--d0-mode learned --seed 7 --report-every 200 "
             "--diffusion 1 --mask-warmup 10000 --mask-scale 0.3 --t-min 0.1 "
             "--optimizer sgd --backend cpu --deterministic "
             "--grad-norm-log %s --loss-log %s "
             "> " BLT_DEVNULL " 2>&1",
             bin_path, log1, loss1);
    snprintf(cmd2, sizeof(cmd2),
             "%s --corpus data/train.bin --steps 200 --lr 0.05 "
             "--block-size 4 --window 48 --embed 64 --hidden 128 --layers 2 "
             "--d0-mode learned --seed 7 --report-every 200 "
             "--diffusion 1 --mask-warmup 10000 --mask-scale 0.3 --t-min 0.1 "
             "--optimizer sgd --backend cpu --deterministic "
             "--grad-norm-log %s --loss-log %s "
             "> " BLT_DEVNULL " 2>&1",
             bin_path, log2, loss2);

    int rc1 = system(cmd1);
    int rc2 = system(cmd2);

    // Both runs must succeed
    TEST_ASSERT(rc1 == 0);
    TEST_ASSERT(rc2 == 0);

    // Compare gradnorm logs byte-for-byte
    FILE *f1 = fopen(log1, "rb");
    FILE *f2 = fopen(log2, "rb");
    TEST_ASSERT(f1 != NULL);
    TEST_ASSERT(f2 != NULL);

    fseek(f1, 0, SEEK_END);
    long sz1 = ftell(f1);
    fseek(f2, 0, SEEK_END);
    long sz2 = ftell(f2);
    TEST_ASSERT(sz1 == sz2);

    rewind(f1);
    rewind(f2);
    char *buf1 = (char *)malloc((size_t)sz1);
    char *buf2 = (char *)malloc((size_t)sz2);
    TEST_ASSERT(buf1 != NULL && buf2 != NULL);
    size_t r1 = fread(buf1, 1, (size_t)sz1, f1);
    size_t r2 = fread(buf2, 1, (size_t)sz2, f2);
    fclose(f1);
    fclose(f2);
    TEST_ASSERT(r1 == (size_t)sz1 && r2 == (size_t)sz2);

    int match = (memcmp(buf1, buf2, (size_t)sz1) == 0);
    free(buf1);
    free(buf2);

    // Compare loss logs as well
    FILE *l1 = fopen(loss1, "rb");
    FILE *l2 = fopen(loss2, "rb");
    TEST_ASSERT(l1 != NULL);
    TEST_ASSERT(l2 != NULL);

    fseek(l1, 0, SEEK_END);
    long lsz1 = ftell(l1);
    fseek(l2, 0, SEEK_END);
    long lsz2 = ftell(l2);
    TEST_ASSERT(lsz1 == lsz2);

    rewind(l1);
    rewind(l2);
    char *lbuf1 = (char *)malloc((size_t)lsz1);
    char *lbuf2 = (char *)malloc((size_t)lsz2);
    TEST_ASSERT(lbuf1 != NULL && lbuf2 != NULL);
    size_t lr1 = fread(lbuf1, 1, (size_t)lsz1, l1);
    size_t lr2 = fread(lbuf2, 1, (size_t)lsz2, l2);
    fclose(l1);
    fclose(l2);
    TEST_ASSERT(lr1 == (size_t)lsz1 && lr2 == (size_t)lsz2);

    int loss_match = (memcmp(lbuf1, lbuf2, (size_t)lsz1) == 0);
    free(lbuf1);
    free(lbuf2);

    // Cleanup temp files
    remove(log1);
    remove(log2);
    remove(loss1);
    remove(loss2);

    if (!match) {
        fprintf(stderr, "  [FAIL] gradnorm logs differ between two --deterministic runs\n");
        return 0;
    }
    if (!loss_match) {
        fprintf(stderr, "  [FAIL] loss logs differ between two --deterministic runs\n");
        return 0;
    }

    return 1;
}
