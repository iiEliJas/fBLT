# Profiling fBLT

## CPU

Set `BLT_TRAIN_TIMING=1` to print `[TIMING]` every 50 steps:

```sh
BLT_TRAIN_TIMING=1 ./build/train_blt_d --corpus data/train.bin --steps 200 ...
```

The line reports `fwd`, `bwd`, `opt`, and `step` milliseconds averaged over the
window. Use `perf` for function-level detail:

```sh
perf record -g ./build/train_blt_d ...
perf report
```

## CUDA

Use Nsight Systems via the wrapper:

```sh
scripts/profile_train.sh build-cuda --corpus data/train.bin --steps 50 --backend cuda
```

Then inspect:

```sh
nsys stats --report cuda_gpu_kern_sum runs/profiles/train_*.nsys-rep
```

For a single kernel, use Nsight Compute:

```sh
ncu --set full ./build-cuda/train_blt_d --backend cuda ...
```

For correctness under the profiler, run under compute-sanitizer:

```sh
cmake --build build-cuda --target cuda-sanitize
```

## Reading `[TIMING]`

`[TIMING]` prints every 50 steps and averages the window; it is not a per-step
percentile. Fields: `fwd` (forward), `bwd` (backward), `opt` (optimizer),
`step` (total).

## Benchmark targets

- `bench-infer` — inference throughput (greedy, speculative)
- `bench-cuda` — CUDA/CPU micro/meso/macro benchmark
- `bench-harness` — benchmark harness

## Notes

- Profiling is Linux-only. `*.nsys-rep` and `*.sqlite` are gitignored.
- The trainer does not emit NVTX ranges; CUDA and OS runtime traces still show
  kernel and API time.