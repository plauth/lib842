// Tests that decompressing an invalid bitstream fails
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include "test_util.h"

static const uint8_t INVALID_BITSTREAM[32] = {
    0xdc, 0x86, 0x04, 0x22, 0x39, 0x67, 0x26, 0xa7,
    0xa6, 0x93, 0x2a, 0xe5, 0x5f, 0x74, 0x50, 0x39,
    0x86, 0x4c, 0xba, 0x62, 0x55, 0xf3, 0xfb, 0xe0,
    0x8f, 0x61, 0xe8, 0x14, 0x8c, 0x77, 0xc2, 0x77
};

int main(int argc, char *argv[]) {
    const struct test842_impl *impl;
    if (argc != 2 ||
        (impl = test842_get_impl_by_name(argv[1])) == NULL) {
        printf("test_decompress_crcerror IMPL\n");
        return EXIT_FAILURE;
    }

    uint8_t out[sizeof(INVALID_BITSTREAM)*2];
    size_t olen = sizeof(out);
    if (impl->decompress(INVALID_BITSTREAM, sizeof(INVALID_BITSTREAM), out, &olen) != -EINVAL) {
        printf("Decompression should have failed with EINVAL\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
