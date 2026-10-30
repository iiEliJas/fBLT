def rng_next(state):
    state[0] = (state[0] + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state[0]
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return (z ^ (z >> 31)) & ((1 << 64) - 1)


def rng_uniform01(state):
    return (rng_next(state) >> 40) / (1 << 24)


def train_window_index(step, num_windows, seed):
    if num_windows <= 1:
        return 0

    def mix64(z):
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    epoch = step // num_windows
    j = step % num_windows
    s = mix64(seed + 0x9E3779B97F4A7C15 * (epoch + 1))
    b = mix64(s) % num_windows
    a = mix64(s ^ 0xD1B54A32D192ED03) % num_windows
    if a == 0:
        a = 1
    tries = 0
    while tries < 64 and gcd(a, num_windows) != 1:
        a = num_windows - 1 if a <= 1 else a - 1
        tries += 1
    if gcd(a, num_windows) != 1:
        a = 1
    return (a * j + b) % num_windows
