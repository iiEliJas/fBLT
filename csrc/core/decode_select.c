#include "core/decode_select.h"

#include "core/backend.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double uniform01(uint64_t *s) { return (double)(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0); }

int blt_decode_is_sampling(const blt_decode_options *o) {
    if (!o) return 0;
    return o->temperature > 0.0f || o->top_p > 0.0f || o->repeat_penalty != 1.0f || o->no_repeat_ngram >= 2;
}

static uint8_t argmax_byte(const float *logits, size_t vocab_size) {
    size_t best = 0;
    float best_val = logits[0];
    for (size_t v = 1; v < vocab_size; v++) {
        if (logits[v] > best_val) {
            best_val = logits[v];
            best = v;
        }
    }
    return (uint8_t)best;
}

// Mark every byte that would complete an n-gram already present in history.
// With order N, the last N-1 bytes are the context; every earlier occurrence of
// that context bans the byte that followed it there.
static void ban_repeat_ngram(size_t order, uint8_t *banned, const uint8_t *history, size_t history_len) {
    if (history_len < order) return;
    const size_t ctx = order - 1;

    for (size_t start = 0; start + order <= history_len; start++) {
        size_t k = 0;
        while (k < ctx && history[start + k] == history[history_len - ctx + k]) k++;
        if (k == ctx) banned[history[start + order - 1]] = 1;
    }
}

uint8_t blt_decode_select(const blt_decode_options *o, const float *logits, size_t vocab_size, const uint8_t *history,
                          size_t history_len, uint64_t *rng) {
    if (!blt_decode_is_sampling(o)) return argmax_byte(logits, vocab_size);

    uint8_t *banned = (uint8_t *)calloc(vocab_size, 1);
    BLT_REQUIRE(banned != NULL, "blt_decode_select: ban table alloc failed");

    if (o->no_repeat_ngram >= 2 && history_len >= o->no_repeat_ngram) {
        ban_repeat_ngram(o->no_repeat_ngram, banned, history, history_len);
    }

    // A ban can cover every byte when the history is shorter than the context
    // plus one, or when the model is fully committed to a loop. Dropping the
    // bans is better than having nothing left to sample from.
    size_t n_banned = 0;
    for (size_t v = 0; v < vocab_size; v++) n_banned += banned[v];
    if (n_banned >= vocab_size) memset(banned, 0, vocab_size);

    float *work = (float *)malloc(vocab_size * sizeof(float));
    BLT_REQUIRE(work != NULL, "blt_decode_select: logits staging alloc failed");

    const float penalty = o->repeat_penalty;
    uint8_t seen[256] = {0};
    const size_t scan_from = (history_len > 256) ? history_len - 256 : 0;
    for (size_t i = scan_from; i < history_len; i++) seen[history[i]] = 1;

    for (size_t v = 0; v < vocab_size; v++) {
        float x = logits[v];
        if (seen[v] && penalty != 1.0f) {
            // Penalise in log space so the penalty is symmetric regardless of
            // the token's sign, which is the usual convention.
            x = (x > 0.0f) ? x / penalty : x * penalty;
        }
        work[v] = x;
    }
    free(banned);

    // temperature <= 0 means "no rescaling". That is distinct from the greedy
    // fast path: the penalty and the ngram ban can pull a run onto this path on
    // their own, and dividing by zero here would turn every logit into inf/NaN.
    const float inv_t = (o->temperature > 0.0f) ? 1.0f / o->temperature : 1.0f;
    for (size_t v = 0; v < vocab_size; v++) work[v] *= inv_t;

    // softmax, with the max shifted out to keep exp() in range
    float maxv = work[0];
    for (size_t v = 1; v < vocab_size; v++) {
        if (work[v] > maxv) maxv = work[v];
    }
    double sum = 0.0;
    for (size_t v = 0; v < vocab_size; v++) {
        work[v] = expf(work[v] - maxv);
        sum += work[v];
    }
    const float inv_sum = (sum > 0.0) ? (float)(1.0 / sum) : 0.0f;
    for (size_t v = 0; v < vocab_size; v++) work[v] *= inv_sum;

    if (o->top_p > 0.0f && o->top_p < 1.0f) {
        size_t *order = (size_t *)malloc(vocab_size * sizeof(size_t));
        BLT_REQUIRE(order != NULL, "blt_decode_select: nucleus alloc failed");
        for (size_t v = 0; v < vocab_size; v++) order[v] = v;
        // Insertion sort on the probabilities keyed through order: vocab is 256,
        // so this beats anything fancier and keeps ties in index order.
        for (size_t i = 1; i < vocab_size; i++) {
            const size_t key = order[i];
            const float pk = work[key];
            size_t j = i;
            while (j > 0 && work[order[j - 1]] < pk) {
                order[j] = order[j - 1];
                j--;
            }
            order[j] = key;
        }
        double cum = 0.0;
        size_t nucleus = vocab_size - 1;
        for (size_t i = 0; i < vocab_size; i++) {
            cum += work[order[i]];
            if (cum >= (double)o->top_p) {
                nucleus = i;
                break;
            }
        }
        double r = uniform01(rng);
        double acc = 0.0;
        for (size_t i = 0; i <= nucleus; i++) {
            acc += work[order[i]];
            if ((float)r < (float)acc) {
                const uint8_t pick = (uint8_t)order[i];
                free(order);
                free(work);
                return pick;
            }
        }
        const uint8_t pick = (uint8_t)order[nucleus];
        free(order);
        free(work);
        return pick;
    }

    const double r = uniform01(rng);
    double acc = 0.0;
    for (size_t v = 0; v < vocab_size; v++) {
        acc += work[v];
        if ((float)r < (float)acc) {
            free(work);
            return (uint8_t)v;
        }
    }
    free(work);
    return (uint8_t)(vocab_size - 1);
}