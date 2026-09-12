#include "blt/infer/stats.h"
#include <stdio.h>

void blt_infer_stats_reset(blt_infer_stats *stats) {
    stats->nfe_encoder_global = 0;
    stats->nfe_decoder = 0;
    stats->bytes_drafted = 0;
    stats->bytes_accepted = 0;
}

float blt_infer_stats_acceptance_rate(const blt_infer_stats *stats) {
    if (stats->bytes_drafted == 0) {
        return 0.0f;
    }
    return (float)stats->bytes_accepted / (float)stats->bytes_drafted;
}

void blt_infer_stats_print(const blt_infer_stats *stats, const char *label) {
    printf("%s: nfe_enc_glob=%zu, nfe_dec=%zu, drafted=%zu, accepted=%zu (%.1f%%)\n", label, stats->nfe_encoder_global,
           stats->nfe_decoder, stats->bytes_drafted, stats->bytes_accepted,
           (double)(100.0f * blt_infer_stats_acceptance_rate(stats)));
}
