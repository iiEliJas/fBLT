#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "test_helpers.h"
#include "test_suite.h"

#include "train_eval.h"

#define TOPK_N 256

// Independent reference ordering: stable insertion sort of class indices by
// descending logit, so exact ties resolve to the lower class index.
static void ref_topk(const float *row, int n, int *idx) {
    for (int i = 0; i < n; i++) idx[i] = i;
    for (int i = 1; i < n; i++) {
        const int cur = idx[i];
        int j = i - 1;
        while (j >= 0 && row[idx[j]] < row[cur]) {
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = cur;
    }
}

static int matches_ref(const float *row, int n, int k, const int *got) {
    int want[TOPK_N];
    ref_topk(row, n, want);
    for (int i = 0; i < k; i++) {
        const int expected = (i < n) ? want[i] : -1;
        if (got[i] != expected) return 0;
    }
    return 1;
}

// Regression for top-5 having counted ranks 6-10 instead of 1-5: the target
// class must be found in slots 0-4, and the 5th-best class must sit in slot 4.
static int run_topk_rank_window_tests(void) {
    float row[TOPK_N];
    for (int v = 0; v < TOPK_N; v++) row[v] = (float)v;
    // Ramp ascending: class 0 is worst, class 255 is best.
    int idx[10];
    topk_from_logits(row, TOPK_N, 10, idx);

    TEST_ASSERT(idx[0] == 255);
    TEST_ASSERT(idx[4] == 251);
    TEST_ASSERT(idx[9] == 246);

    // Mirror how the tools accumulate: top-1 reads slot 0, top-5 scans slots
    // 0-4, top-10 scans slots 0-9. On this ramp 251 is the 5th best, 250 the
    // 6th, and 246 the 10th, so the rank-5/rank-6 boundary is pinned exactly.
    int rank5 = 0, rank6 = 0, rank10 = 0;
    for (int k = 0; k < 10; k++) {
        if (k < 5 && idx[k] == 251) rank5 = 1;
        if (k < 5 && idx[k] == 250) rank6 = 1;
        if (idx[k] == 246) rank10 = 1;
    }
    TEST_ASSERT(rank5);
    TEST_ASSERT(!rank6);
    TEST_ASSERT(rank10);
    return 1;
}

// Exact ties: the max is shared by classes 3 and 250, the runner-up by 7 and 3+4.
// Stable order must put the lower class index first in every tied pair.
static int run_topk_tie_tests(void) {
    float row[TOPK_N];
    for (int v = 0; v < TOPK_N; v++) row[v] = -1.0f;
    row[3] = 4.0f;
    row[7] = 4.0f;
    row[100] = 9.0f;
    row[250] = 9.0f;

    int idx[10];
    topk_from_logits(row, TOPK_N, 10, idx);
    const int want[10] = {100, 250, 3, 7, 0, 1, 2, 4, 5, 6};
    for (int i = 0; i < 10; i++) TEST_ASSERT(idx[i] == want[i]);

    // A row where every class is identical is the degenerate tie case.
    for (int v = 0; v < TOPK_N; v++) row[v] = 0.5f;
    topk_from_logits(row, TOPK_N, 10, idx);
    for (int i = 0; i < 10; i++) TEST_ASSERT(idx[i] == i);
    return 1;
}

// A row with no positive logit at all. The old inlined selector seeded its
// threshold at 0, so nothing here could ever enter the list.
static int run_topk_all_negative_tests(void) {
    float row[TOPK_N];
    for (int v = 0; v < TOPK_N; v++) row[v] = -1.0f - (float)v;

    int idx[10];
    topk_from_logits(row, TOPK_N, 10, idx);
    for (int i = 0; i < 10; i++) {
        TEST_ASSERT(idx[i] == i);
        TEST_ASSERT(idx[i] != -1);
    }

    // The threshold must not bias the ranking either: a large negative argmax
    // still wins slot 0.
    for (int v = 0; v < TOPK_N; v++) row[v] = -100.0f;
    row[42] = -0.25f;
    topk_from_logits(row, TOPK_N, 10, idx);
    TEST_ASSERT(idx[0] == 42);
    return 1;
}

// Argmax at the final vocab slot. Reading the wrong end of a descending list
// is exactly what made the old top-1 column report 10th-place accuracy.
static int run_topk_argmax_last_tests(void) {
    float row[TOPK_N];
    for (int v = 0; v < TOPK_N; v++) row[v] = -3.0f;
    row[TOPK_N - 1] = 12.0f;

    int idx[10];
    topk_from_logits(row, TOPK_N, 10, idx);
    TEST_ASSERT(idx[0] == TOPK_N - 1);
    TEST_ASSERT(idx[0] != idx[9]);

    // Same for the last index of a short row.
    const float small[4] = {0.5f, 0.5f, 0.5f, 9.0f};
    topk_from_logits(small, 4, 3, idx);
    TEST_ASSERT(idx[0] == 3);
    TEST_ASSERT(idx[1] == 0);
    TEST_ASSERT(idx[2] == 1);
    return 1;
}

// k larger than the row, k of 1, and the degenerate k/n cases.
static int run_topk_edge_cases(void) {
    const float row[4] = {3.0f, 1.0f, 4.0f, 2.0f};

    int idx[6];
    topk_from_logits(row, 4, 6, idx);
    const int want[6] = {2, 0, 3, 1, -1, -1};
    for (int i = 0; i < 6; i++) TEST_ASSERT(idx[i] == want[i]);

    topk_from_logits(row, 4, 1, idx);
    TEST_ASSERT(idx[0] == 2);

    int empty[3] = {7, 7, 7};
    topk_from_logits(row, 0, 3, empty);
    for (int i = 0; i < 3; i++) TEST_ASSERT(empty[i] == -1);

    // A non-positive k must not touch the caller's buffer.
    int untouched[2] = {11, 13};
    topk_from_logits(row, 4, 0, untouched);
    topk_from_logits(row, 4, -3, untouched);
    TEST_ASSERT(untouched[0] == 11);
    TEST_ASSERT(untouched[1] == 13);
    return 1;
}

// Differential check against the reference over duplicate-heavy random rows, so
// the tie-breaking rule is exercised far more than a hand-built case can.
static int run_topk_random_tests(void) {
    float row[TOPK_N];
    int idx[16];
    uint32_t rng = 12345u;
    const int ks[4] = {1, 5, 10, 16};

    for (int trial = 0; trial < 64; trial++) {
        for (int v = 0; v < TOPK_N; v++) {
            rng = rng * 1664525u + 1013904223u;
            // A 4-value range forces frequent exact ties.
            row[v] = -2.0f + (float)((rng >> 16) % 4);
        }
        for (int ki = 0; ki < 4; ki++) {
            const int k = ks[ki];
            topk_from_logits(row, TOPK_N, k, idx);
            if (!matches_ref(row, TOPK_N, k, idx)) {
                fprintf(stderr, "  [FAIL] topk mismatch: trial=%d k=%d\n", trial, k);
                return 0;
            }
            for (int i = 1; i < k; i++) {
                if (row[idx[i - 1]] < row[idx[i]]) {
                    fprintf(stderr, "  [FAIL] topk not descending: trial=%d k=%d slot=%d\n", trial, k, i);
                    return 0;
                }
            }
        }
    }
    return 1;
}

int run_topk_from_logits_tests(void) {
    if (!run_topk_rank_window_tests()) return 0;
    if (!run_topk_tie_tests()) return 0;
    if (!run_topk_all_negative_tests()) return 0;
    if (!run_topk_argmax_last_tests()) return 0;
    if (!run_topk_edge_cases()) return 0;
    if (!run_topk_random_tests()) return 0;
    return 1;
}
