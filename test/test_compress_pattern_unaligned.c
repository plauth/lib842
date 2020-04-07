// Tests the case where the input and output buffers are unaligned
// (not aligned to a 8-byte boundary)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdalign.h>
#include "test_patterns.h"
#include "test_util.h"

int main(int argc, char *argv[]) {
    const struct test842_impl *impl;
    const struct test842_pattern *pattern;
    if (argc != 3 ||
        (impl = test842_get_impl_by_name(argv[1])) == NULL ||
                (pattern = test842_get_pattern_by_name(argv[2])) == NULL) {
        printf("test_compress_pattern_unaligned IMPL PATTERN\n");
        return EXIT_FAILURE;
    }

    alignas(8) uint8_t inb[pattern->uncompressed_len+3], outb[pattern->ref_compressed_len+3];
    uint8_t *in = inb + 3, *out = outb + 3;
    memcpy(in, pattern->uncompressed, pattern->uncompressed_len);
    size_t olen = pattern->ref_compressed_len;
    if (impl->compress(in, pattern->uncompressed_len, out, &olen) != 0) {
        printf("Compression failed\n");
        return EXIT_FAILURE;
    }

    if (olen != pattern->ref_compressed_len ||
        memcmp(out, pattern->ref_compressed, pattern->ref_compressed_len) != 0) {
        printf("Invalid compression result\n");
        printf("Input (%zu bytes):\n", pattern->uncompressed_len);
        test842_hexdump(pattern->uncompressed, pattern->uncompressed_len);
        printf("Expected output (%zu bytes):\n", pattern->ref_compressed_len);
        test842_hexdump(pattern->ref_compressed, pattern->ref_compressed_len);
        printf("Actual output (%zu bytes):\n", olen);
        test842_hexdump(out, olen);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
