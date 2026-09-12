#ifndef BLT_INFER_STATS_H
#define BLT_INFER_STATS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

// Inference forward-pass counters (BLT-S / BLT-DV). Controllers bump these
// at the call site; ops and models stay untouched so training numerics are
// never affected.
//
// NFE = number of function evaluations (forward passes) of a submodule.
typedef struct {
    size_t nfe_encoder_global; // encoder + global transformer invocations (the expensive tier)
    size_t nfe_decoder;        // decoder-only invocations (drafts, verification decodes)
    size_t bytes_drafted;      // speculative bytes produced
    size_t bytes_accepted;     // committed bytes from drafting (includes forced progress)
} blt_infer_stats;

// Zeroes all counters.
void blt_infer_stats_reset(blt_infer_stats *stats);

// bytes_accepted / bytes_drafted; 0.0f when nothing was drafted.
float blt_infer_stats_acceptance_rate(const blt_infer_stats *stats);

// One line to stdout: "label: nfe_enc_glob=.., nfe_dec=.., drafted=.., accepted=.. (..%)"
void blt_infer_stats_print(const blt_infer_stats *stats, const char *label);

#ifdef __cplusplus
}
#endif

#endif // BLT_INFER_STATS_H
