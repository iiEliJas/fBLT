#ifndef BLT_BENCH_HARNESS_H
#define BLT_BENCH_HARNESS_H

#include <stddef.h>
#include <stdint.h>

// Benchmark harness — shared, backend-agnostic measurement infrastructure.
//
// Used by ablation sweeps (BPB tables), patcher performance, and other benchmarks.
// Latency samples are collected in seconds (double); the raw timer works in
// nanoseconds via clock_gettime(CLOCK_MONOTONIC)

// ---- Timer ----------------------------------------------------------------
// Returns elapsed nanoseconds since bench_timer_start() was last called.
void bench_timer_start(void);
uint64_t bench_timer_stop_ns(void);

// Convenience: seconds since bench_timer_start().
double bench_timer_stop_sec(void);

// ---- Statistics ------------------------------------------------------------
typedef struct {
    size_t n; // number of samples used
    double mean;
    double stddev; // population stddev, seconds
    double min;
    double max;
    double p50; // nearest-rank percentile
    double p90;
    double p99;
} bench_stats; // all in seconds

// Compute stats from a raw sample array. Samples must be >= 1 in count.
// Callers are expected to discard cold-cache/warmup samples before passing
// the array in (see bench_collect_samples below).
void bench_stats_compute(bench_stats *out, const double *samples, size_t n);

// ---- Domain metrics (flat key:value float map) ------------------------------
#define BENCH_MAX_METRICS 16
#define BENCH_KEY_LEN 32
#define BENCH_NAME_LEN 64

typedef struct {
    char key[BENCH_KEY_LEN];
    double value;
} bench_kv;

// ---- Result ------------------------------------------------------------------
#define BENCH_TAG_LEN 64

typedef struct {
    char name[BENCH_NAME_LEN]; // e.g. "patcher_global"
    char tag[BENCH_TAG_LEN];   // config tag, e.g. "ngram_345_v100k"
    char phase[BENCH_TAG_LEN]; // e.g. "5.1_ngram"
    bench_stats latency;       // wall-clock per iteration, seconds
    bench_kv metrics[BENCH_MAX_METRICS];
    size_t num_metrics;
} bench_result;

void bench_result_init(bench_result *r, const char *name, const char *tag, const char *phase);

// Add/overwrite a domain metric (BPB, NFEs, GB/s, flops_per_byte, ...).
void bench_result_add_metric(bench_result *r, const char *key, double value);

// ---- Collection helper ---------------------------------------------------------
// Runs `fn(arg)` `warmup + iterations` times. The first run is a cold-cache
// pass whose sample is discarded; warmup runs are also discarded. The
// remaining `iterations` latency samples (seconds) are written to `samples`,
// which must hold at least `iterations` doubles.
// Plan rule of thumb: iterations >= 20 for stable mean/p99.
typedef void (*bench_fn)(void *arg);
void bench_collect_samples(bench_fn fn, void *arg, size_t warmup, size_t iterations, double *samples);

// ---- JSONL output ------------------------------------------------------------------
// Appends exactly one JSON line:
// {"timestamp":<unix_s>,"phase":"...","name":"...","tag":"...",
//  "latency":{"n":..,"mean":..,"stddev":..,"min":..,"max":..,
//             "p50":..,"p90":..,"p99":..},
//  "metrics":{"bpb":..,...}}
// Latency values are in seconds. Line-delimited so partial/killed runs never
// corrupt history and results can be tailed/grep'd directly.
// Returns 0 on success, -1 on I/O failure.
int bench_write_json(const char *path, const bench_result *r);

#endif
