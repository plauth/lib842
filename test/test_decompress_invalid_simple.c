// Tests that decompressing an invalid bitstream fails (simple test with a fixed bitstream)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "test_util.h"

static const uint8_t INVALID_BITSTREAM[] = {
	0xdc, 0x86, 0x04, 0x22, 0x39, 0x67, 0x26, 0xa7, 0xa6, 0x93, 0x2a,
	0xe5, 0x5f, 0x74, 0x50, 0x39, 0x86, 0x4c, 0xba, 0x62, 0x55, 0xf3,
	0xfb, 0xe0, 0x8f, 0x61, 0xe8, 0x14, 0x8c, 0x77, 0xc2, 0x77
};

int main(int argc, char *argv[])
{
	const struct lib842_implementation *impl;
	if (argc != 2 || (impl = test842_get_impl_by_name(argv[1])) == NULL) {
		printf("test_decompress_invalid_simple IMPL\n");
		return EXIT_FAILURE;
	}


	uint8_t *in = aligned_alloc(impl->required_alignment, sizeof(INVALID_BITSTREAM));
	memcpy(in, INVALID_BITSTREAM, sizeof(INVALID_BITSTREAM));
	size_t olen = sizeof(INVALID_BITSTREAM) * 2;
	uint8_t *out = aligned_alloc(impl->required_alignment, olen);
	if (impl->decompress(in, sizeof(INVALID_BITSTREAM), out, &olen) != -EINVAL) {
		printf("Decompression should have failed with EINVAL\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
