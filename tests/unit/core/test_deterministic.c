#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "test_helpers.h"

// Regression test: run the training binary twice with --deterministic and
// assert the gradnorm logs are byte-identical. This catches any CUDA
// nondeterminism reintroduced by future changes.

int run_deterministic_regression_test(void) {
    // Skip if the training binary or corpus is not available
    if (access("./bin/train_blt_d", X_OK) != 0) {
        fprintf(stderr, "  [SKIP] bin/train_blt_d not built; skipping deterministic test\n");
        return 1;
    }
    if (access("data/train.bin", R_OK) != 0) {
        fprintf(stderr, "  [SKIP] data/train.bin not found; skipping deterministic test\n");
        return 1;
    }

    // Use a small config: 200 steps, seed=7, deterministic
    const char *cmd1 = "./bin/train_blt_d --corpus data/train.bin --steps 200 --lr 0.05 "
                       "--block-size 4 --window 48 --embed 64 --hidden 128 --layers 2 "
                       "--d0-mode learned --seed 7 --report-every 200 "
                       "--diffusion 1 --mask-warmup 10000 --mask-scale 0.3 --t-min 0.1 "
                       "--optimizer sgd --backend cpu --deterministic "
                       "--grad-norm-log /tmp/blt_det_test1.log --loss-log /tmp/blt_det_loss1.log "
                       "> /dev/null 2>&1";

    const char *cmd2 = "./bin/train_blt_d --corpus data/train.bin --steps 200 --lr 0.05 "
                       "--block-size 4 --window 48 --embed 64 --hidden 128 --layers 2 "
                       "--d0-mode learned --seed 7 --report-every 200 "
                       "--diffusion 1 --mask-warmup 10000 --mask-scale 0.3 --t-min 0.1 "
                       "--optimizer sgd --backend cpu --deterministic "
                       "--grad-norm-log /tmp/blt_det_test2.log --loss-log /tmp/blt_det_loss2.log "
                       "> /dev/null 2>&1";

    int rc1 = system(cmd1);
    int rc2 = system(cmd2);

    // Both runs must succeed
    TEST_ASSERT(rc1 == 0);
    TEST_ASSERT(rc2 == 0);

    // Compare gradnorm logs byte-for-byte
    FILE *f1 = fopen("/tmp/blt_det_test1.log", "rb");
    FILE *f2 = fopen("/tmp/blt_det_test2.log", "rb");
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
    FILE *l1 = fopen("/tmp/blt_det_loss1.log", "rb");
    FILE *l2 = fopen("/tmp/blt_det_loss2.log", "rb");
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
    remove("/tmp/blt_det_test1.log");
    remove("/tmp/blt_det_test2.log");
    remove("/tmp/blt_det_loss1.log");
    remove("/tmp/blt_det_loss2.log");

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
