#include "blt/models/entropy.h"
#include "blt/core/backend.h"
#include "blt/ops/row_stats.h"

void blt_compute_entropy(const blt_tensor *probs, blt_tensor *entropy_out, const blt_entropy_config *config) {
    BLT_REQUIRE(config != NULL, "blt_compute_entropy: config cannot be NULL");

    blt_check_nd_fp32(probs, 2, (const size_t[]){0, config->vocab_size},
                      "Input probabilities tensor must be 2D with last dimension equal to vocab_size");

    blt_check_nd_fp32(
        entropy_out, 1, (const size_t[]){probs->shape[0]},
        "Output entropy tensor must be 1D with shape matching the number of rows in input probabilities tensor");

    blt_entropy_rows(probs, entropy_out, config->use_log2 ? 1 : 0);
}
