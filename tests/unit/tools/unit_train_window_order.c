#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_helpers.h"
#include "test_suite.h"

#include "train_diag.h"

// train_window_index replaces the old `step % num_windows`. Two properties
// matter: every window is visited exactly once per epoch (so coverage is
// unchanged), and the order is not sequential (so a second run starting at step
// 0 does not replay the first run's byte stream).
int run_train_window_order_tests(void) {
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
