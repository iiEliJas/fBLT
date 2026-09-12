#ifndef BLT_BENCH_TIMER_H
#define BLT_BENCH_TIMER_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#include <profileapi.h>
#else
#include <time.h>
#endif

typedef struct {
    double min_time;
    double max_time;
    double avg_time;
    double total_time;
    size_t iterations;
} blt_bench_stats;

typedef struct {
#ifdef _WIN32
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    LARGE_INTEGER frequency;
#else
    struct timespec start;
    struct timespec end;
#endif
} blt_timer;

static inline void blt_timer_init(blt_timer *timer) {
#ifdef _WIN32
    QueryPerformanceFrequency(&timer->frequency);
    timer->start.QuadPart = 0;
    timer->end.QuadPart = 0;
#else
    timer->start.tv_sec = 0;
    timer->start.tv_nsec = 0;
    timer->end.tv_sec = 0;
    timer->end.tv_nsec = 0;
#endif
}

static inline void blt_timer_start(blt_timer *timer) {
#ifdef _WIN32
    QueryPerformanceCounter(&timer->start);
#else
    clock_gettime(CLOCK_MONOTONIC, &timer->start);
#endif
}

static inline void blt_timer_stop(blt_timer *timer) {
#ifdef _WIN32
    QueryPerformanceCounter(&timer->end);
#else
    clock_gettime(CLOCK_MONOTONIC, &timer->end);
#endif
}

static inline double blt_timer_elapsed(const blt_timer *timer) {
#ifdef _WIN32
    LONGLONG elapsed = timer->end.QuadPart - timer->start.QuadPart;
    return (double)elapsed / (double)timer->frequency.QuadPart;
#else
    time_t sec_diff = timer->end.tv_sec - timer->start.tv_sec;
    long nsec_diff = timer->end.tv_nsec - timer->start.tv_nsec;
    return (double)sec_diff + (double)nsec_diff / 1e9;
#endif
}

static inline void blt_timer_reset(blt_timer *timer) {
#ifdef _WIN32
    timer->start.QuadPart = 0;
    timer->end.QuadPart = 0;
#else
    timer->start.tv_sec = 0;
    timer->start.tv_nsec = 0;
    timer->end.tv_sec = 0;
    timer->end.tv_nsec = 0;
#endif
}

static inline double blt_throughput_gb_per_sec(size_t bytes, double time_sec) {
    if (time_sec <= 0.0) return 0.0;
    return (double)bytes / time_sec / 1e9;
}

static inline blt_bench_stats blt_bench_run(void (*benchmark_fn)(void), size_t iterations, size_t warmup_iterations) {
    blt_timer timer;
    blt_timer_init(&timer);

    for (size_t i = 0; i < warmup_iterations; ++i) {
        benchmark_fn();
    }

    double *times = (double *)malloc(iterations * sizeof(double));
    if (!times) {
        blt_bench_stats empty = {0};
        return empty;
    }

    for (size_t i = 0; i < iterations; ++i) {
        blt_timer_start(&timer);
        benchmark_fn();
        blt_timer_stop(&timer);
        times[i] = blt_timer_elapsed(&timer);
    }

    double min_time = times[0];
    double max_time = times[0];
    double total_time = 0.0;

    for (size_t i = 0; i < iterations; ++i) {
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
        total_time += times[i];
    }

    double avg_time = total_time / iterations;

    blt_bench_stats stats = {min_time = min_time, max_time = max_time, avg_time = avg_time, total_time = total_time,
                             iterations = iterations};

    free(times);
    return stats;
}

#endif