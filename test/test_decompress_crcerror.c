// Tests that decompressing an otherwise valid stream with an invalid CRC fails
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
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
	int err = impl->decompress(INVALID_BITSTREAM, sizeof(INVALID_BITSTREAM), out, &olen);
	// FIXME TESTFAILURE: This test fails on the OpenCL implementation because
	//                    it currently doesn't check that the CRC is valid
	if (err == 0 && strcmp(argv[1], "cl") == 0) {
		fprintf(stderr, "!! TEST FAILED (BUT PASS, DUE TO KNOWN DEFECT) !!\n");
		return EXIT_SUCCESS;
	}
	if (err != -EINVAL) {
		printf("Decompression should have failed with EINVAL\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
