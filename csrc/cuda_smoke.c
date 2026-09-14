#include <stdio.h>

extern int blt_cuda_smoke_run(void);

int main(void) {
    int rc = blt_cuda_smoke_run();
    if (rc == 0) {
        printf("[cuda-smoke] PASS\n");
    } else {
        fprintf(stderr, "[cuda-smoke] FAIL (rc=%d)\n", rc);
    }
    return rc;
}
