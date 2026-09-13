#include "harness.h"

#include "core/backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Timer

static struct timespec bench_ts_start;

void bench_timer_start(void) { clock_gettime(CLOCK_MONOTONIC, &bench_ts_start); }

uint64_t bench_timer_stop_ns(void) {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    uint64_t sec = (uint64_t)(end.tv_sec - bench_ts_start.tv_sec);
    long nsec = end.tv_nsec - bench_ts_start.tv_nsec;
    return sec * 1000000000ULL + (uint64_t)nsec;
}

double bench_timer_stop_sec(void) { return (double)bench_timer_stop_ns() / 1e9; }

// Statistics

static int bench_cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

// Nearest-rank percentile on a sorted array of size n (n >= 1)
static double bench_percentile_sorted(const double *sorted, size_t n, double q) {
    double rank = (q / 100.0) * (double)n;
    size_t idx = (size_t)ceil(rank);
    if (idx == 0) idx = 1;
    if (idx > n) idx = n;
    return sorted[idx - 1];
}

void bench_stats_compute(bench_stats *out, const double *samples, size_t n) {
    memset(out, 0, sizeof(*out));
    if (n == 0 || samples == NULL) return;

    out->n = n;

    double total = 0.0;
    double min = samples[0];
    double max = samples[0];
    for (size_t i = 0; i < n; ++i) {
        total += samples[i];
        if (samples[i] < min) min = samples[i];
        if (samples[i] > max) max = samples[i];
    }
    out->mean = total / (double)n;
    out->min = min;
    out->max = max;

    double sq_total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = samples[i] - out->mean;
        sq_total += d * d;
    }
    out->stddev = sqrt(sq_total / (double)n);

    double *sorted = (double *)malloc(n * sizeof(double));
    if (sorted == NULL) {
        BLT_FATAL("bench_stats_compute: out of memory");
    }
    memcpy(sorted, samples, n * sizeof(double));
    qsort(sorted, n, sizeof(double), bench_cmp_double);
    out->p50 = bench_percentile_sorted(sorted, n, 50.0);
    out->p90 = bench_percentile_sorted(sorted, n, 90.0);
    out->p99 = bench_percentile_sorted(sorted, n, 99.0);
    free(sorted);
}

// Result

static void bench_copy_str(char *dst, size_t cap, const char *src) {
    if (src == NULL) src = "";
    strncpy(dst, src, cap - 1);
    dst[cap - 1] = '\0';
}

void bench_result_init(bench_result *r, const char *name, const char *tag, const char *phase) {
    memset(r, 0, sizeof(*r));
    bench_copy_str(r->name, BENCH_NAME_LEN, name);
    bench_copy_str(r->tag, BENCH_TAG_LEN, tag);
    bench_copy_str(r->phase, BENCH_TAG_LEN, phase);
}

void bench_result_add_metric(bench_result *r, const char *key, double value) {
    for (size_t i = 0; i < r->num_metrics; ++i) {
        if (strncmp(r->metrics[i].key, key, BENCH_KEY_LEN) == 0) {
            r->metrics[i].value = value;
            return;
        }
    }
    if (r->num_metrics >= BENCH_MAX_METRICS) {
        BLT_FATAL("bench_result_add_metric: metric map full (%d)", BENCH_MAX_METRICS);
    }
    bench_copy_str(r->metrics[r->num_metrics].key, BENCH_KEY_LEN, key);
    r->metrics[r->num_metrics].value = value;
    ++r->num_metrics;
}

// Collection

void bench_collect_samples(bench_fn fn, void *arg, size_t warmup, size_t iterations, double *samples) {
    bench_timer_start();
    fn(arg);
    bench_timer_stop_ns();

    for (size_t i = 0; i < warmup; ++i) {
        fn(arg);
    }

    for (size_t i = 0; i < iterations; ++i) {
        bench_timer_start();
        fn(arg);
        samples[i] = bench_timer_stop_sec();
    }
}

// JSONL output

int bench_write_json(const char *path, const bench_result *r) {
    FILE *f = fopen(path, "a");
    if (f == NULL) return -1;

    time_t now = time(NULL);

    if (fprintf(f, "{\"timestamp\":%lld,\"phase\":\"%s\",\"name\":\"%s\",\"tag\":\"%s\",", (long long)now, r->phase,
                r->name, r->tag) < 0) {
        fclose(f);
        return -1;
    }

    if (fprintf(f,
                "\"latency\":{\"n\":%zu,\"mean\":%.9e,\"stddev\":%.9e,"
                "\"min\":%.9e,\"max\":%.9e,\"p50\":%.9e,\"p90\":%.9e,\"p99\":%.9e}",
                r->latency.n, r->latency.mean, r->latency.stddev, r->latency.min, r->latency.max, r->latency.p50,
                r->latency.p90, r->latency.p99) < 0) {
        fclose(f);
        return -1;
    }

    if (fputc(',', f) == EOF) {
        fclose(f);
        return -1;
    }

    if (fprintf(f, "\"metrics\":{") < 0) {
        fclose(f);
        return -1;
    }
    for (size_t i = 0; i < r->num_metrics; ++i) {
        const char *sep = (i > 0) ? "," : "";
        if (fprintf(f, "%s\"%s\":%.9e", sep, r->metrics[i].key, r->metrics[i].value) < 0) {
            fclose(f);
            return -1;
        }
    }
    if (fprintf(f, "}}\n") < 0) {
        fclose(f);
        return -1;
    }

    if (fclose(f) != 0) return -1;
    return 0;
}
