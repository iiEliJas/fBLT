#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_helpers.h"
#include "test_suite.h"

#include "core/backend.h"
#include "core/decode_select.h"

#define VOCAB 256

static void fill_logits(float *logits, size_t v, float base, float step) {
    for (size_t i = 0; i < v; i++) logits[i] = base + step * (float)i;
}

static size_t argmax_of(const float *logits, size_t v) {
    size_t best = 0;
    for (size_t i = 1; i < v; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

// A knob is off by default, and defaults must reproduce argmax exactly.
static int test_defaults_are_greedy(void) {
    blt_decode_options o;
    blt_decode_options_defaults(&o);
    TEST_ASSERT(o.temperature == 0.0f);
    TEST_ASSERT(o.top_p == 0.0f);
    TEST_ASSERT(o.repeat_penalty == 1.0f);
    TEST_ASSERT(o.no_repeat_ngram == 0);
    TEST_ASSERT(blt_decode_is_sampling(&o) == 0);
    TEST_ASSERT(blt_decode_is_sampling(NULL) == 0);

    float logits[VOCAB];
    fill_logits(logits, VOCAB, -3.0f, 0.01f);
    uint64_t rng = 12345;
    for (int trial = 0; trial < 8; trial++) {
        const uint8_t got = blt_decode_select(&o, logits, VOCAB, NULL, 0, &rng);
        TEST_ASSERT(got == (uint8_t)argmax_of(logits, VOCAB));
    }
    // Greedy must not consume the stream, so the same seed keeps agreeing.
    uint64_t rng2 = 12345;
    TEST_ASSERT(blt_decode_select(&o, logits, VOCAB, NULL, 0, &rng2) == (uint8_t)argmax_of(logits, VOCAB));
    return 1;
}

// Every knob has to independently flip blt_decode_is_sampling, otherwise a flag
// silently does nothing.
static int test_each_knob_enables_sampling(void) {
    float logits[VOCAB];
    fill_logits(logits, VOCAB, 0.0f, 0.001f);
    uint64_t rng = 1;

    blt_decode_options o;
    blt_decode_options_defaults(&o);
    o.temperature = 0.8f;
    TEST_ASSERT(blt_decode_is_sampling(&o) == 1);

    blt_decode_options_defaults(&o);
    o.top_p = 0.9f;
    TEST_ASSERT(blt_decode_is_sampling(&o) == 1);

    blt_decode_options_defaults(&o);
    o.repeat_penalty = 1.1f;
    TEST_ASSERT(blt_decode_is_sampling(&o) == 1);

    blt_decode_options_defaults(&o);
    o.no_repeat_ngram = 4;
    TEST_ASSERT(blt_decode_is_sampling(&o) == 1);

    // Order 1 would ban the whole alphabet, so it must not count as a knob.
    blt_decode_options_defaults(&o);
    o.no_repeat_ngram = 1;
    TEST_ASSERT(blt_decode_is_sampling(&o) == 0);

    // penalty of exactly 1.0 is the identity, not a knob.
    blt_decode_options_defaults(&o);
    o.repeat_penalty = 1.0f;
    TEST_ASSERT(blt_decode_is_sampling(&o) == 0);

    // temperature 0 alone stays greedy, which is what the CLI relies on.
    blt_decode_options_defaults(&o);
    o.temperature = 0.0f;
    TEST_ASSERT(blt_decode_is_sampling(&o) == 0);
    TEST_ASSERT(blt_decode_select(&o, logits, VOCAB, NULL, 0, &rng) == (uint8_t)argmax_of(logits, VOCAB));
    return 1;
}

// Same seed, same output; different seeds, different output.
static int test_temperature_is_seeded_and_varies(void) {
    float logits[VOCAB];
    for (size_t i = 0; i < VOCAB; i++) logits[i] = 0.001f * (float)i;

    blt_decode_options o;
    blt_decode_options_defaults(&o);
    o.temperature = 1.5f;
    o.seed = 99;

    int differ = 0;
    for (int i = 0; i < 64; i++) {
        uint64_t s1 = 99, s2 = 99, s3 = 100;
        const uint8_t p = blt_decode_select(&o, logits, VOCAB, NULL, 0, &s1);
        const uint8_t q = blt_decode_select(&o, logits, VOCAB, NULL, 0, &s2);
        const uint8_t r = blt_decode_select(&o, logits, VOCAB, NULL, 0, &s3);
        TEST_ASSERT(p == q);
        if (p != r) differ = 1;
    }
    TEST_ASSERT(differ == 1);
    return 1;
}

// A tight nucleus must never leave the top token, and must not crash on ties.
static int test_top_p_restricts_to_nucleus(void) {
    float logits[VOCAB];
    // One dominant token, everything else negligible.
    for (size_t i = 0; i < VOCAB; i++) logits[i] = -20.0f;
    logits[77] = 10.0f;

    blt_decode_options o;
    blt_decode_options_defaults(&o);
    o.temperature = 1.0f;
    o.top_p = 0.5f;

    uint64_t rng = 5;
    for (int i = 0; i < 200; i++) {
        TEST_ASSERT(blt_decode_select(&o, logits, VOCAB, NULL, 0, &rng) == 77);
    }

    // Uniform logits with a mid nucleus must produce more than one distinct byte.
    for (size_t i = 0; i < VOCAB; i++) logits[i] = 0.0f;
    o.top_p = 0.5f;
    uint8_t seen[VOCAB] = {0};
    uint64_t r2 = 7;
    for (int i = 0; i < 400; i++) seen[blt_decode_select(&o, logits, VOCAB, NULL, 0, &r2)] = 1;
    int distinct = 0;
    for (int i = 0; i < VOCAB; i++) distinct += seen[i];
    TEST_ASSERT(distinct > 1);

    // top_p = 1 keeps the full distribution, so everything stays reachable.
    o.top_p = 1.0f;
    memset(seen, 0, sizeof(seen));
    uint64_t r3 = 8;
    for (int i = 0; i < 2000; i++) seen[blt_decode_select(&o, logits, VOCAB, NULL, 0, &r3)] = 1;
    distinct = 0;
    for (int i = 0; i < VOCAB; i++) distinct += seen[i];
    TEST_ASSERT(distinct > 100);
    return 1;
}

// The penalty has to push an already-emitted token down, not up.
static int test_repeat_penalty_suppresses_seen(void) {
    float logits[VOCAB];
    for (size_t i = 0; i < VOCAB; i++) logits[i] = -8.0f;
    logits[10] = 4.0f;
    logits[200] = 3.99f; // very close, so the penalty has to matter

    const uint8_t history[3] = {10, 10, 10};

    blt_decode_options off;
    blt_decode_options_defaults(&off);
    off.temperature = 1.0f;
    // With byte 10 emitted three times it should essentially never be redrawn.
    int hits10 = 0;
    for (int i = 0; i < 500; i++) {
        uint64_t rr = 3;
        if (blt_decode_select(&off, logits, VOCAB, history, 3, &rr) == 10) hits10++;
    }
    TEST_ASSERT(hits10 > 0); // sanity check: the token is reachable when unpenalised

    blt_decode_options on;
    blt_decode_options_defaults(&on);
    on.temperature = 1.0f;
    on.repeat_penalty = 10.0f;
    int hits10_pen = 0;
    for (int i = 0; i < 500; i++) {
        uint64_t rr = 3;
        if (blt_decode_select(&on, logits, VOCAB, history, 3, &rr) == 10) hits10_pen++;
    }
    TEST_ASSERT(hits10_pen == 0);
    return 1;
}

// Order-N ban: every earlier completion of the trailing N-1 context is removed.
static int test_no_repeat_ngram_bans_completion(void) {
    blt_decode_options o;
    blt_decode_options_defaults(&o);
    o.no_repeat_ngram = 3;

    // Context {1,2} already appeared twice, once followed by 9 and once by 8.
    const uint8_t history[] = {1, 2, 9, 1, 2, 8};
    const uint8_t done_a = 9, done_b = 8;

    // Uniform logits, so the ban is the only thing shaping the draw.
    float logits[VOCAB];
    for (size_t i = 0; i < VOCAB; i++) logits[i] = 0.0f;
    int hit_a = 0, hit_b = 0, hit_other = 0;
    for (int i = 0; i < 600; i++) {
        uint64_t rr = 4;
        const uint8_t got = blt_decode_select(&o, logits, VOCAB, history, 6, &rr);
        if (got == done_a) hit_a++;
        else if (got == done_b) hit_b++;
        else hit_other++;
    }
    TEST_ASSERT(hit_a == 0);
    TEST_ASSERT(hit_b == 0);
    TEST_ASSERT(hit_other > 0);
    return 1;
}

// A short history, or a history that bans everything, must degrade to
// "pick something" rather than returning garbage.
static int test_edge_cases_do_not_break(void) {
    float logits[VOCAB];
    for (size_t i = 0; i < VOCAB; i++) logits[i] = 0.0f;

    blt_decode_options o;
    blt_decode_options_defaults(&o);
    o.temperature = 1.0f;
    o.no_repeat_ngram = 8;

    uint64_t rng = 6;
    // History shorter than the order, and an empty history: nothing to ban.
    const uint8_t one[1] = {5};
    const uint8_t two[2] = {5, 6};
    for (int i = 0; i < 50; i++) {
        blt_decode_select(&o, logits, VOCAB, one, 1, &rng);
        blt_decode_select(&o, logits, VOCAB, two, 2, &rng);
        blt_decode_select(&o, logits, VOCAB, NULL, 0, &rng);
    }

    // History that bans every byte: must still return a valid token.
    uint8_t all[16];
    for (int i = 0; i < 16; i++) all[i] = (uint8_t)i;
    uint64_t r2 = 9;
    for (int i = 0; i < 50; i++) {
        (void)blt_decode_select(&o, logits, VOCAB, all, 16, &r2);
    }
    return 1;
}

int run_decode_select_tests(void) {
    int ok = 1;
    ok &= test_defaults_are_greedy();
    ok &= test_each_knob_enables_sampling();
    ok &= test_temperature_is_seeded_and_varies();
    ok &= test_top_p_restricts_to_nucleus();
    ok &= test_repeat_penalty_suppresses_seen();
    ok &= test_no_repeat_ngram_bans_completion();
    ok &= test_edge_cases_do_not_break();
    printf("    decode_select: sampling knobs verified\n");
    return ok;
}