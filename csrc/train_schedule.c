#include "train_schedule.h"

#include <math.h>

float train_lr_at_step(const args_t *a, size_t step, size_t end_step) {
    const float min_lr = (a->min_lr >= 0.0f) ? a->min_lr : 0.1f * a->lr;
    if (a->warmup_steps > 0 && step < a->warmup_steps) {
        return a->lr * (float)(step + 1) / (float)(a->warmup_steps);
    }
    if (a->lr_schedule == 1) {
        const size_t start = a->warmup_steps;
        const size_t span = (end_step > start) ? (end_step - start) : 1;
        double progress = 0.0;
        if (step > start) {
            progress = (double)(step - start) / (double)span;
        }
        if (progress > 1.0) progress = 1.0;
        if (progress < 0.0) progress = 0.0;
        const double pi = 3.14159265358979323846;
        const double coeff = 0.5 * (1.0 + cos(pi * progress));
        return (float)(min_lr + coeff * (double)(a->lr - min_lr));
    }
    float lr = a->lr;
    if (a->lr_decay_count > 0) {
        for (int d = 0; d < a->lr_decay_count; d++)
            if (step >= a->lr_decay_steps[d]) lr *= a->lr_decay_factor;
    } else if (a->lr_decay) {
        if (step >= (end_step * 85) / 100) lr *= 0.09f;
        else if (step >= (end_step * 60) / 100) lr *= 0.3f;
    }
    return lr;
}