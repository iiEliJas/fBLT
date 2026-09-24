#ifndef BLT_TRAIN_ARGS_H
#define BLT_TRAIN_ARGS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define BLT_TRAIN_LR_DECAY_MAX 16

typedef struct {
    const char *corpus_path;
    size_t steps;
    float lr;
    size_t block_size;
    size_t window;
    size_t embed;
    size_t hidden;
    size_t layers;
    int d0_learned;
    uint64_t seed;
    size_t report_every;
    int diffusion;
    const char *eval_path;
    size_t eval_windows;
    const char *save_path;
    size_t save_every;
    const char *load_path;
    size_t eval_skip;
    float t_min;
    int lr_decay;
    size_t lr_decay_steps[BLT_TRAIN_LR_DECAY_MAX];
    int lr_decay_count;
    float lr_decay_factor;
    size_t mask_warmup;
    float mask_scale;
    float last_row_scale;
    const char *train_entlm;
    const char *entropy_lm;
    size_t mask_late_step;
    float mask_late_scale;
    int entropy_patches;
    int use_cuda;
    size_t enc_layers;
    size_t glob_layers;
    size_t dec_layers;
    int cross_last;
    float t_warmup_frac;
    float t_hi_start;
    int optimizer;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float max_norm;
    const char *grad_norm_log;
    const char *update_norm_log;
    const char *component_norm_log;
    const char *activation_dump_log;
    const char *batch_log;
    size_t eval_every;
    const char *loss_log;
    int deterministic;
    size_t cuda_scratch_mb;
    size_t model_mb;
} args_t;

// Parse command-line arguments into args_t with default values.
// Calls exit(1) on error (same as original usage() behavior).
args_t parse_args(int argc, char **argv);

#ifdef __cplusplus
}
#endif
#endif // BLT_TRAIN_ARGS_H
