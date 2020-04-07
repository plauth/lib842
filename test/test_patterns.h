#ifndef LIB842_TEST_DATA_H
#define LIB842_TEST_DATA_H

#include <stdint.h>
#include <stddef.h>

// Pair of (uncompressed data, compressed data) to be used as a source for tests
struct test842_pattern {
    // Uncompressed data
    const uint8_t *uncompressed;
    size_t uncompressed_len;

    // Possible compressed data (note, that there is no 'canonical' compressed
    // form; every compressor implementation can generate different but valid
    // bitstreams for the same input)
    const uint8_t *compressed;
    size_t compressed_len;
};

const struct test842_pattern *test842_get_pattern_by_name(const char *name);

#endif
