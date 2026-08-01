#ifndef BLT_CORE_BACKEND_H
#define BLT_CORE_BACKEND_H

#include <stdio.h>
#include <stdlib.h>

#define BLT_FATAL(msg) do { \
    fprintf(stderr, "BLT Fatal Error [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
    exit(EXIT_FAILURE); \
} while (0)

#endif // BLT_CORE_BACKEND_H
