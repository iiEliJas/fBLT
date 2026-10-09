// Byte-level decode-time sampling controls.
//
// These are pure functions over one row of logits plus the bytes generated so
// far, with no model or backend dependency, so they unit-test in isolation.
//
// Order of operations matters and matches common practice:
//   1. apply no-repeat-ngram bans
//   2. apply the repeat penalty to already-seen tokens
//   3. divide logits by the temperature
//   4. softmax
//   5. keep the smallest prefix whose mass reaches top_p
//   6. draw from that prefix
//
// Doing the bans before the penalty keeps the ban absolute: a token that is
// banned cannot come back because penalising other tokens changed the softmax.
#ifndef BLT_CORE_DECODE_SELECT_H
#define BLT_CORE_DECODE_SELECT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef struct {
    float temperature;      // <= 0 selects greedy argmax and skips steps 2-6
    float top_p;            // <= 0 disables nucleus truncation
    float repeat_penalty;   // < 1.0 disables; applies to tokens already emitted
    size_t no_repeat_ngram; // 0 disables; byte-level n-gram order, must be >= 2
    uint64_t seed;          // seeds the sampling stream
} blt_decode_options;

// All fields off: plain argmax. This is the default so callers that never opt in
// keep exactly the previous behaviour.
static inline void blt_decode_options_defaults(blt_decode_options *o) {
    o->temperature = 0.0f;
    o->top_p = 0.0f;
    o->repeat_penalty = 1.0f;
    o->no_repeat_ngram = 0;
    o->seed = 11;
}

// True when any knob that changes the chosen byte is enabled.
int blt_decode_is_sampling(const blt_decode_options *o);

// Pick the next byte from `logits` [vocab_size] given `history` [history_len].
//
// history must include the prompt: the repeat penalty and the ngram ban both
// operate on everything emitted so far, not just the generated tail. rng must
// be a live splitmix64 state; it is advanced only when sampling actually runs,
// so greedy decoding leaves the stream untouched.
uint8_t blt_decode_select(const blt_decode_options *o, const float *logits, size_t vocab_size, const uint8_t *history,
                          size_t history_len, uint64_t *rng);

#ifdef __cplusplus
}
#endif
#endif // BLT_CORE_DECODE_SELECT_H