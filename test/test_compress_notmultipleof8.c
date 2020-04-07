// Tests that compressing an amount of data that is not a multiple of 8 fails
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include "test_util.h"

static uint8_t NOT_MULTIPLE_OF_8[] = { 0x11, 0x11, 0x11, 0x12, 0x12};

int main(int argc, char *argv[]) {
    const struct test842_impl *impl;
    if (argc != 2 ||
        (impl = test842_get_impl_by_name(argv[1])) == NULL) {
        printf("test_compress_notmultipleof8 IMPL\n");
        return EXIT_FAILURE;
    }

    uint8_t out[32];
    size_t olen = sizeof(out);
    if (impl->compress(NOT_MULTIPLE_OF_8, sizeof(NOT_MULTIPLE_OF_8), out, &olen) != -EINVAL) {
        printf("Compression should have failed with EINVAL\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
