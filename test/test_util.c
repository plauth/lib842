#include "test_util.h"
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sw842.h>
#include <hw842.h>

#define HEXDUMP_BYTES_PER_LINE 8

void test842_hexdump(const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i += HEXDUMP_BYTES_PER_LINE) {
        for (size_t j = i; j < i + HEXDUMP_BYTES_PER_LINE && j < len; j++) {
            printf("0x%.2x", data[j]);
            if (j != len - 1) {
                printf(", ");
            }
        }
        printf("\n");
    }
}

static const struct test842_impl IMPL_SW = {
        .compress = sw842_compress,
        .decompress = sw842_decompress
};

static const struct test842_impl IMPL_OPTSW = {
        .compress = optsw842_compress,
        .decompress = optsw842_decompress
};

#ifdef LIB842_HAVE_CRYPTODEV_LINUX_COMP
static const struct test842_impl IMPL_HW = {
        .compress = hw842_compress,
        .decompress = hw842_decompress
};
#endif

const struct test842_impl *test842_get_impl_by_name(const char *name) {
    if (strcmp(name, "sw") == 0) {
        return &IMPL_SW;
    } else if (strcmp(name, "optsw") == 0) {
        return &IMPL_OPTSW;
    #ifdef LIB842_HAVE_CRYPTODEV_LINUX_COMP
    } else if (strcmp(name, "hw") == 0) {
        return &IMPL_HW;
    #endif
    }
    return NULL;
}
