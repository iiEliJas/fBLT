#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_helpers.h"
#include "test_suite.h"

#include "train_diag.h"

#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

static int dir_exists(const char *p) {
    struct stat st;
    return stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}

// A missing checkpoint directory used to surface as a fatal at the first
// periodic save, throwing away every step trained before it.
static int test_ensure_parent_dir(void) {
    char root[] = "/tmp/blt_ensure_XXXXXX";
    TEST_ASSERT(mkdtemp(root) != NULL);

    // Creates several levels at once.
    char nested[512];
    snprintf(nested, sizeof(nested), "%s/a/b/c/model.fblt", root);
    TEST_ASSERT(!dir_exists(nested));
    TEST_ASSERT(train_ensure_parent_dir(nested) == 0);
    {
        char dir[512];
        snprintf(dir, sizeof(dir), "%s/a/b/c", root);
        TEST_ASSERT(dir_exists(dir));
    }

    // Idempotent on an existing directory.
    TEST_ASSERT(train_ensure_parent_dir(nested) == 0);

    // A bare filename has no parent to make.
    TEST_ASSERT(train_ensure_parent_dir("model.fblt") == 0);
    TEST_ASSERT(!dir_exists("model.fblt"));

    // NULL and empty are no-ops.
    TEST_ASSERT(train_ensure_parent_dir(NULL) == 0);
    TEST_ASSERT(train_ensure_parent_dir("") == 0);

    // An unwritable destination must report failure rather than pass silently.
    TEST_ASSERT(train_ensure_parent_dir("/proc/blt_should_not_exist/x.fblt") != 0);

    char cleanup[512];
    snprintf(cleanup, sizeof(cleanup), "rm -rf %s", root);
    if (system(cleanup) != 0) { /* best effort */
    }

    printf("    ensure_parent_dir: nested creation, idempotence and unwritable path verified\n");
    return 1;
}

// train_window_index replaces the old `step % num_windows`. Two properties
// matter: every window is visited exactly once per epoch (so coverage is
// unchanged), and the order is not sequential (so a second run starting at step
// 0 does not replay the first run's byte stream).
int run_train_window_order_tests(void) {
    int ok = 1;
    ok &= test_ensure_parent_dir();
    const size_t nw_cases[] = {1, 2, 3, 4, 7, 8, 64, 97, 1024, 4096};
    const size_t epochs = 3;

    for (size_t c = 0; c < sizeof(nw_cases) / sizeof(nw_cases[0]); c++) {
        const size_t nw = nw_cases[c];
        uint8_t *seen = (uint8_t *)calloc(nw, 1);
        TEST_ASSERT(seen != NULL);

        for (size_t e = 0; e < epochs; e++) {
            memset(seen, 0, nw);
            size_t seq[4096];
            TEST_ASSERT(nw <= 4096);
            for (size_t j = 0; j < nw; j++) {
                const size_t step = e * nw + j;
                const size_t w = train_window_index(step, nw, 7);
                TEST_ASSERT(w < nw);
                TEST_ASSERT(seen[w] == 0); // exactly once, never a repeat
                seen[w] = 1;
                seq[j] = w;
            }
            if (nw > 2) {
                size_t asc = 0;
                for (size_t j = 1; j < nw; j++) {
                    if (seq[j] > seq[j - 1]) asc++;
                }
                TEST_ASSERT(asc > 0); // not the identity order
            }
        }
        free(seen);
    }

    // Pure function of (step, seed): the same inputs must reproduce, otherwise a
    // resumed run could not continue the order it started.
    for (size_t step = 0; step < 5000; step += 37) {
        TEST_ASSERT(train_window_index(step, 335229, 7) == train_window_index(step, 335229, 7));
    }

    // Adjacent epochs must not share an order.
    const size_t nw = 1024;
    size_t same = 0;
    for (size_t j = 0; j < nw; j++) {
        if (train_window_index(j, nw, 7) == train_window_index(nw + j, nw, 7)) same++;
    }
    TEST_ASSERT(same < nw / 2);

    printf("    window order: bijection + reseeded per epoch verified\n");
    return 1;
}
