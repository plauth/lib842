// Tests that decompressing an otherwise valid stream with an invalid CRC fails
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include "test_util.h"

static const uint8_t INVALID_BITSTREAM[] = {
	0x00, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x04, 0x84, 0x84,
	0x84, 0x84, 0x84, 0x84, 0x84, 0xbd, 0x42, 0xdf, 0xe3, 0x00, 0x00, 0x00
};

int main(int argc, char *argv[])
{
	const struct test842_impl *impl;
	if (argc != 2 || (impl = test842_get_impl_by_name(argv[1])) == NULL) {
		printf("test_decompress_crcerror IMPL\n");
		return EXIT_FAILURE;
	}

	uint8_t out[sizeof(INVALID_BITSTREAM) * 2];
	size_t olen = sizeof(out);
	if (impl->decompress(INVALID_BITSTREAM, sizeof(INVALID_BITSTREAM), out,
			     &olen) != -EINVAL) {
		printf("Decompression should have failed with EINVAL\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
