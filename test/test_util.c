#include "test_util.h"
#include <stdio.h>
#include <string.h>
#include <lib842/sw.h>
#include <lib842/hw.h>
#include <lib842/cl.h>

#define HEXDUMP_BYTES_PER_LINE 8

void test842_hexdump(const uint8_t *data, size_t len)
{
	for (size_t i = 0; i < len; i += HEXDUMP_BYTES_PER_LINE) {
		for (size_t j = i; j < i + HEXDUMP_BYTES_PER_LINE && j < len;
		     j++) {
			printf("0x%.2x", data[j]);
			if (j != len - 1) {
				printf(", ");
			}
		}
		printf("\n");
	}
}

#ifdef LIB842_HAVE_OPENCL
static const struct lib842_implementation IMPL_CL = { // Test mock
	.decompress = cl842_decompress,
	.required_alignment = 1
};
#endif

const struct lib842_implementation *test842_get_impl_by_name(const char *name)
{
	if (strcmp(name, "sw") == 0)
		return get_sw842_implementation();
	if (strcmp(name, "optsw") == 0)
		return get_optsw842_implementation();
#ifdef LIB842_HAVE_CRYPTODEV_LINUX_COMP
	if (strcmp(name, "hw") == 0)
		return get_hw842_implementation();
#endif
#ifdef LIB842_HAVE_OPENCL
	if (strcmp(name, "cl") == 0)
		return &IMPL_CL;
#endif
	return NULL;
}
