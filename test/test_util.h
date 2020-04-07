#ifndef LIB842_TEST_UTIL_H
#define LIB842_TEST_UTIL_H

#include <stdint.h>
#include <stddef.h>

// Print an array of bytes as hexadecimal to stdout
void test842_hexdump(const uint8_t *data, size_t len);

// Gets the specified implementation (e.g. software, hardware) or the (de)compressor
struct test842_impl {
    int (*compress)(const uint8_t *in, size_t ilen, uint8_t *out, size_t *olen);
    int (*decompress)(const uint8_t *in, size_t ilen, uint8_t *out, size_t *olen);

};
const struct test842_impl *test842_get_impl_by_name(const char *name);

#endif //LIB842_TEST_UTIL_H
