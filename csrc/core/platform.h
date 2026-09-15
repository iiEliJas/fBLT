// Cross-platform abstractions for Windows/POSIX port
//
//   blt_time_sec() monotonic wall-clock seconds (QPC on Win32, clock_gettime on POSIX)
//   blt_get_temp_dir() platform temp directory path
//   BLT_ACCESS / BLT_F_OK / BLT_R_OK / BLT_X_OK portable access() shim
//   BLT_DEVNULL is "/dev/null" or "NUL"

#ifndef BLT_PLATFORM_H
#define BLT_PLATFORM_H

#ifdef _WIN32

// -- Windows ------------------------
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <direct.h>

// access() shim — _access() on MSVC, same signature.
#define BLT_ACCESS _access
#define BLT_F_OK 0
#define BLT_R_OK 4
#define BLT_X_OK 0 // exec-bit meaningless on Windows; 0 = file exists
#define BLT_DEVNULL "NUL"

static inline double blt_time_sec(void) {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
}

static inline int blt_get_temp_dir(char *buf, size_t len) {
    DWORD n = GetTempPathA((DWORD)len, buf);
    return (n > 0 && n < len) ? 0 : -1;
}

#else

// -- POSIX (Linux, macOS, BSDs) ------------------------
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define BLT_ACCESS access
#define BLT_F_OK F_OK
#define BLT_R_OK R_OK
#define BLT_X_OK X_OK
#define BLT_DEVNULL "/dev/null"

static inline double blt_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static inline int blt_get_temp_dir(char *buf, size_t len) {
    const char *tmp = getenv("TMPDIR");
    if (!tmp || tmp[0] == '\0') tmp = "/tmp";
    size_t n = strlen(tmp);
    if (n >= len) return -1;
    memcpy(buf, tmp, n + 1);
    return 0;
}

#endif // _WIN32

#endif // BLT_PLATFORM_H
