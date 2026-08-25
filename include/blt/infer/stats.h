#ifndef BLT_INFER_STATS_H
#define BLT_INFER_STATS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>


// Forward-pass counters for the inference controllers (BLT-S /
// BLT-DV). Controllers increment these at the call site; ops and models
// stay unmodified so training numerics are never affected.
//
// NFE = number of function evaluations (forward passes) of a submodule.
typedef struct {
    size_t nfe_encoder_global;  // encode-stage invocations (encoder + global transformer; the expensive tier)
    size_t nfe_decoder;         // decoder-only invocations (drafting steps, verification decodes)
    size_t bytes_drafted;       // speculative bytes produced by a drafting policy
    size_t bytes_accepted;      // committed bytes that came from a drafting policy (incl. forced progress)
} blt_infer_stats;


// Zeroes all counters.
void blt_infer_stats_reset(blt_infer_stats* stats);

// bytes_accepted / bytes_drafted; returns 0.0f when nothing was drafted.
float blt_infer_stats_acceptance_rate(const blt_infer_stats* stats);

// Prints one line "label: nfe_enc_glob=.., nfe_dec=.., drafted=.., accepted=.. (..%)"
// to stdout.
void blt_infer_stats_print(const blt_infer_stats* stats, const char* label);

#ifdef __cplusplus
}
#endif

#endif // BLT_INFER_STATS_H
