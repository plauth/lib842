#ifndef LIB842_TEST_TEST_UTIL_H
#define LIB842_TEST_TEST_UTIL_H

#include <lib842/common.h>
#include <stdint.h>
#include <stddef.h>

// Print an array of bytes as hexadecimal to stdout
void test842_hexdump(const uint8_t *data, size_t len);

// Gets the specified implementation (e.g. software, hardware) or the (de)compressor
const struct lib842_implementation *test842_get_impl_by_name(const char *name);

#endif // LIB842_TEST_TEST_UTIL_H
