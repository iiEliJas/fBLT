#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_helpers.h"
#include "test_suite.h"

#include "models/patcher.h"

// A patch array and the capacity handed to the segmenter must be sized from the
// same constant. Getting that wrong writes past the end of a stack array, and
// the mismatch is invisible until a window is long enough to overflow it, so
// check the source text instead of waiting for a crash.
int run_patch_capacity_consistency_tests(void) {
    TEST_ASSERT(BLT_MAX_PATCHES >= 512);

    static const char *sources[] = {
        "csrc/train_blt_d.c",
        "csrc/infer/block_generation.c",
    };
    int checked = 0;

    for (size_t f = 0; f < sizeof(sources) / sizeof(sources[0]); f++) {
        FILE *fp = fopen(sources[f], "rb");
        if (!fp) {
            printf("    cannot open %s (run tests from the repo root)\n", sources[f]);
            return 0;
        }
        char buf[1 << 20];
        const size_t got = fread(buf, 1, sizeof(buf) - 1, fp);
        fclose(fp);
        buf[got] = '\0';

        // No caller may declare a patch array with its own size.
        TEST_ASSERT(strstr(buf, "blt_patch_info") != NULL);
        for (const char *p = strstr(buf, "blt_patch_info"); p != NULL; p = strstr(p + 1, "blt_patch_info")) {
            const char *open_br = strchr(p, '[');
            if (!open_br || (size_t)(open_br - p) >= 64) continue; // a type reference, not a declaration
            static const char kPrefix[] = "blt_patch_info ";
            static const char kWant[] = "[BLT_MAX_PATCHES]";
            if (strncmp(p, kPrefix, sizeof(kPrefix) - 1) != 0) continue;
            if (strncmp(open_br, kWant, sizeof(kWant) - 1) != 0) {
                printf("    %s: patch array must be sized [BLT_MAX_PATCHES], got %.32s\n", sources[f], open_br);
                return 0;
            }
            checked++;
        }
    }

    printf("    patch capacity: %d array sites all sized [BLT_MAX_PATCHES]\n", checked);
    return 1;
}
