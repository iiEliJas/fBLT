#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_helpers.h"
#include "test_suite.h"

#include "train_args.h"
#include "train_schedule.h"

static int test_lr_warmup(void) {
    args_t a;
    memset(&a, 0, sizeof(a));
    a.lr = 1e-4f;
    a.warmup_steps = 10;
    a.lr_schedule = 0;
    a.min_lr = -1.0f;
    a.lr_decay = 0;
    a.lr_decay_count = 0;
    a.lr_decay_factor = 0.3f;

    float l0 = train_lr_at_step(&a, 0, 100);
    float l9 = train_lr_at_step(&a, 9, 100);
    float l10 = train_lr_at_step(&a, 10, 100);
    if (l0 < 9e-6f || l0 > 1.1e-5f) return 0;
    if (l9 < 9.9e-5f || l9 > 1.01e-4f) return 0;
    if (l10 != a.lr) return 0;
    return 1;
}

static int test_lr_cosine_endpoint(void) {
    args_t a;
    memset(&a, 0, sizeof(a));
    a.lr = 1e-4f;
    a.warmup_steps = 0;
    a.lr_schedule = 1;
    a.min_lr = 1e-5f;
    float lend = train_lr_at_step(&a, 100, 100);
    if (lend < 9.9e-6f || lend > 1.01e-5f) return 0;
    return 1;
}

static int test_lr_cosine_min_zero(void) {
    args_t a;
    memset(&a, 0, sizeof(a));
    a.lr = 1e-4f;
    a.warmup_steps = 0;
    a.lr_schedule = 1;
    a.min_lr = 0.0f;
    float lend = train_lr_at_step(&a, 50, 50);
    if (lend != 0.0f) return 0;
    return 1;
}

int run_lr_schedule_tests(void) {
    int ok = 1;
    ok &= test_lr_warmup();
    ok &= test_lr_cosine_endpoint();
    ok &= test_lr_cosine_min_zero();
    return ok;
}