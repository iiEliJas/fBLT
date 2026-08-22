#include "bench/harness.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// Harness self-test — exit criteria from docs/Plan ("Before Phase 5"):
//
//   1. Running the same fixed config twice produces latency stats whose means
//      agree within ~2%.
//   2. Writes two dummy JSONL entries so tools/bench_report.py can be checked
//      against them (delta table).
//
// Run via `make bench-harness`. Exit code 0 = pass.
// ==========================================================================

#define ITERATIONS 30 // plan: every benchmark >= 20 times
#define WARMUP 5

#define SELFTEST_TOLERANCE 0.02 // 2% mean agreement

// Deterministic dummy workload: repeated FMA sweep over a static buffer,
// big enough (~100us) that timer resolution is irrelevant, small enough that
// the whole self-test stays fast. Result funneled through a volatile sink so
// the optimizer cannot elide it.
#define WORKLOAD_ELEMS 65536
static float work_buf[WORKLOAD_ELEMS];
static volatile double work_sink;

static void workload(void* arg) {
    (void)arg;
    double acc = 0.0;
    for (int rep = 0; rep < 8; ++rep) {
        for (size_t i = 0; i < WORKLOAD_ELEMS; ++i) {
            acc += (double)work_buf[i] * 0.5 + 1.0;
        }
    }
    work_sink = acc;
}

static void init_work_buf(void) {
    for (size_t i = 0; i < WORKLOAD_ELEMS; ++i) {
        work_buf[i] = (float)((i * 2654435761u) % 1000) / 1000.0f;
    }
}

// One full measurement pass of the fixed config: collect samples + stats.
static void measure_once(bench_stats* out) {
    double samples[ITERATIONS];
    bench_collect_samples(workload, NULL, WARMUP, ITERATIONS, samples);
    bench_stats_compute(out, samples, ITERATIONS);
}

int main(int argc, char** argv) {
    const char* jsonl_path = (argc > 1) ? argv[1] : "bench/dummy_results.jsonl";

    init_work_buf();

    printf("BLT benchmark harness self-test\n");
    printf("  workload: fixed FMA sweep, %d iterations x%d runs (warmup %d, cold pass discarded)\n",
           ITERATIONS, 2, WARMUP);

    // ---- Criterion 1: two runs of the same config agree within 2% ----
    bench_stats run1;
    bench_stats run2;
    measure_once(&run1);
    measure_once(&run2);

    printf("  run1: mean=%.3f ms stddev=%.3f ms p50=%.3f ms p99=%.3f ms\n",
           run1.mean * 1e3, run1.stddev * 1e3, run1.p50 * 1e3, run1.p99 * 1e3);
    printf("  run2: mean=%.3f ms stddev=%.3f ms p50=%.3f ms p99=%.3f ms\n",
           run2.mean * 1e3, run2.stddev * 1e3, run2.p50 * 1e3, run2.p99 * 1e3);

    double diff = fabs(run1.mean - run2.mean);
    double rel = diff / fmax(run1.mean, run2.mean);

    int pass = 1;
    if (rel > SELFTEST_TOLERANCE) {
        fprintf(stderr, "[FAIL] run means differ by %.2f%% (tolerance %.2f%%)\n",
                rel * 100.0, SELFTEST_TOLERANCE * 100.0);
        pass = 0;
    } else {
        printf("  [OK] run means agree within %.2f%% (<= %.2f%%)\n",
               rel * 100.0, SELFTEST_TOLERANCE * 100.0);
    }

    // Sanity: stats fields are populated and ordered.
    if (!(run1.n == ITERATIONS && run2.n == ITERATIONS)) {
        fprintf(stderr, "[FAIL] sample count wrong\n");
        pass = 0;
    }
    if (!(run1.min <= run1.p50 && run1.p50 <= run1.p90 && run1.p90 <= run1.p99 &&
          run1.p99 <= run1.max)) {
        fprintf(stderr, "[FAIL] percentiles not monotonic\n");
        pass = 0;
    }

    // ---- Criterion 2: emit two dummy JSONL entries for bench_report.py ----
    bench_result r1;
    bench_result_init(&r1, "selftest_workload", "dummy_a", "harness_selftest");
    r1.latency = run1;
    bench_result_add_metric(&r1, "dummy_metric", 42.0);
    bench_result_add_metric(&r1, "gb_per_sec", 1.0 / run1.mean / 1e9 * 8.0 * WORKLOAD_ELEMS * 8);

    bench_result r2;
    bench_result_init(&r2, "selftest_workload", "dummy_b", "harness_selftest");
    r2.latency = run2;
    bench_result_add_metric(&r2, "dummy_metric", 45.0);
    bench_result_add_metric(&r2, "gb_per_sec", 1.0 / run2.mean / 1e9 * 8.0 * WORKLOAD_ELEMS * 8);

    if (bench_write_json(jsonl_path, &r1) != 0 ||
        bench_write_json(jsonl_path, &r2) != 0) {
        fprintf(stderr, "[FAIL] could not append to %s\n", jsonl_path);
        return 1;
    }
    printf("  [OK] wrote 2 dummy entries to %s\n", jsonl_path);
    printf("  check report: python3 tools/bench_report.py --results %s --baseline dummy_a\n",
           jsonl_path);

    printf(pass ? "SELFTEST PASS\n" : "SELFTEST FAIL\n");
    return pass ? 0 : 1;
}
