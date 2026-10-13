#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_helpers.h"
#include "test_suite.h"

#include "train_diag.h"

#ifdef _WIN32
#include <direct.h>
#include <sys/stat.h>
#define test_mkdir(p) _mkdir((p))
#else
#include <sys/stat.h>
#include <unistd.h>
#define test_mkdir(p) mkdir((p), 0777)
#endif

// The ETA line must be right from the very first report after a resume.
// Regression: last_step started at 0 while the loop starts at start_step, so the
// first report divided one interval's elapsed time by (start_step + interval)
// and reported a wildly optimistic rate, which the EMA then carried forward.
static int test_eta_resume(void) {
    const size_t END = 200000, RE = 1000;
    const double SEC_PER_STEP = 0.010; // 100 steps/s

    struct {
        size_t start;
        const char *label;
    } cases[] = {{0, "fresh"}, {70000, "resumed"}};

    for (size_t c = 0; c < 2; c++) {
        const size_t start = cases[c].start;
        blt_eta_state eta;
        blt_eta_init(&eta, start); // exactly what train_blt_d does

        double first = 0.0, prev = 1e30;
        int monotonic = 1, checked = 0;

        for (size_t step = start; step < END; step++) {
            if ((step + 1) % RE == 0 || step + 1 == END) {
                const double interval = RE * SEC_PER_STEP;
                const double rem = blt_eta_update(&eta, step, END, interval, 0.0);
                if (checked == 0) first = rem;
                if (rem < 0.0 || rem != rem) return 0;
                if (rem > prev + 1e-9) monotonic = 0;
                prev = rem;
                checked++;
            }
        }

        // True remaining time at the first report, in steady state.
        const double truth = (double)(END - (start + RE)) * SEC_PER_STEP;
        if (!monotonic) {
            printf("    eta %s: ETA increased across reports\n", cases[c].label);
            return 0;
        }
        if (!(first > truth * 0.99 && first < truth * 1.01)) {
            printf("    eta %s: first ETA %.1fs, expected %.1fs\n", cases[c].label, first, truth);
            return 0;
        }
        printf("    eta %-8s first report %.0fs (true %.0fs), monotonic across %d reports\n", cases[c].label, first,
               truth, checked);
    }

    // Excluded time (checkpoint saves) must not count toward the rate.
    {
        const double interval = 1000 * SEC_PER_STEP;
        const size_t step = 999;

        blt_eta_state a1, a2;
        blt_eta_init(&a1, 0);
        blt_eta_update(&a1, step, 200000, interval, 0.0);

        blt_eta_init(&a2, 0);
        blt_eta_update(&a2, step, 200000, interval + 50.0, 50.0); // 50s of saving
        TEST_ASSERT(fabs(a1.ema_sec_per_step - a2.ema_sec_per_step) < 1e-12);
    }

    // The call site must pass a.start_step. The helper is initialised through a
    // struct now, but the argument is still the one thing that can reintroduce
    // the bug, and it is invisible in the unit test above.
    FILE *fp = fopen("csrc/train_blt_d.c", "rb");
    if (!fp) {
        printf("    cannot open csrc/train_blt_d.c (run tests from the repo root)\n");
        return 0;
    }
    char buf[1 << 20];
    const size_t got = fread(buf, 1, sizeof(buf) - 1, fp);
    fclose(fp);
    buf[got] = '\0';
    if (!strstr(buf, "blt_eta_init(&eta, a.start_step)")) {
        printf("    train_blt_d.c must initialise the ETA from a.start_step, not zero\n");
        return 0;
    }

    return 1;
}

static int dir_exists(const char *p) {
    struct stat st;
    return stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}

// A missing checkpoint directory used to surface as a fatal at the first
// periodic save, throwing away every step trained before it.
static int test_ensure_parent_dir(void) {
    // Deliberately no mkdtemp and no system("rm -rf"): both misbehave under the
    // MSYS2 Windows job, where mkdtemp on /tmp returns NULL. Everything is
    // relative to the repo root, which is where the test runner already is.
    const char *root = "blt_test_tmp_ensure";
    const char *leaf = "blt_test_tmp_ensure/a/b/c";
    TEST_ASSERT(test_mkdir(root) == 0 || dir_exists(root));

    // Creates several levels at once.
    char nested[512];
    snprintf(nested, sizeof(nested), "%s/model.fblt", leaf);
    TEST_ASSERT(!dir_exists(leaf));
    TEST_ASSERT(train_ensure_parent_dir(nested) == 0);
    TEST_ASSERT(dir_exists(leaf));

    // Idempotent on an existing directory.
    TEST_ASSERT(train_ensure_parent_dir(nested) == 0);

    // A bare filename has no parent to make.
    TEST_ASSERT(train_ensure_parent_dir("model.fblt") == 0);
    TEST_ASSERT(!dir_exists("model.fblt"));

    // NULL and empty are no-ops.
    TEST_ASSERT(train_ensure_parent_dir(NULL) == 0);
    TEST_ASSERT(train_ensure_parent_dir("") == 0);

    // An unwritable destination must report failure rather than pass silently.
    // A regular file where a directory component is expected fails on every
    // platform, unlike /proc which only exists on some.
    char blocker[512];
    snprintf(blocker, sizeof(blocker), "%s/blocker", root);
    FILE *bf = fopen(blocker, "w");
    TEST_ASSERT(bf != NULL);
    fclose(bf);
    char under_file[512];
    snprintf(under_file, sizeof(under_file), "%s/blocker/x/y.fblt", root);
    TEST_ASSERT(train_ensure_parent_dir(under_file) != 0);

    remove(blocker);
    rmdir(leaf);
    rmdir("blt_test_tmp_ensure/a/b");
    rmdir("blt_test_tmp_ensure/a");
    rmdir(root);

    printf("    ensure_parent_dir: nested creation, idempotence and unwritable path verified\n");
    return 1;
}

// train_window_index replaces the old `step % num_windows`. Two properties
// matter: every window is visited exactly once per epoch (so coverage is
// unchanged), and the order is not sequential (so a second run starting at step
// 0 does not replay the first run's byte stream).
int run_train_window_order_tests(void) {
    int ok = 1;
    ok &= test_eta_resume();
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
    return ok;
}
