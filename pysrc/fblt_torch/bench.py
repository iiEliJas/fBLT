import time


def benchmark(fn, iters=10):
    t0 = time.time()
    for _ in range(iters):
        fn()
    return (time.time() - t0) / iters
