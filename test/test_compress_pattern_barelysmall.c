// Tests the case where during compression, the output buffer is too small
// (minus one byte, so just barely smaller than required)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "test_patterns.h"
#include "test_util.h"

int main(int argc, char *argv[]) {
    const struct test842_impl *impl;
    const struct test842_pattern *pattern;
    if (argc != 3 ||
        (impl = test842_get_impl_by_name(argv[1])) == NULL ||
        (pattern = test842_get_pattern_by_name(argv[2])) == NULL) {
        printf("test_compress_pattern_barelysmall IMPL PATTERN\n");
        return EXIT_FAILURE;
    }

    uint8_t out[pattern->ref_compressed_len-1];
    size_t olen = pattern->ref_compressed_len-1;
    if (impl->compress(pattern->uncompressed, pattern->uncompressed_len, out, &olen) != -ENOSPC) {
        printf("Compression should have failed with ENOSPC\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
