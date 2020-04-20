// Tests that the result of decompressing some data matches the expected reference output
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdalign.h>
#include "test_patterns.h"
#include "test_util.h"

int main(int argc, char *argv[])
{
	const struct test842_impl *impl;
	const struct test842_pattern *pattern;
	if (argc != 3 || (impl = test842_get_impl_by_name(argv[1])) == NULL ||
	    (pattern = test842_get_pattern_by_name(argv[2])) == NULL) {
		printf("test_decompress_pattern IMPL PATTERN\n");
		return EXIT_FAILURE;
	}

	// Note: We overallocate the output buffer a bit (5 bytes), to make sure
	// the decompressor recovers the correct uncompressed length
	// This also makes the test work on real HW (the nx-842 kernel driver
	// doesn't accept an output buffer of size 0 even if it's sufficient)
	alignas(8) uint8_t in[pattern->compressed_len],
		out[pattern->uncompressed_len + 5];
	memcpy(in, pattern->compressed, pattern->compressed_len);
	size_t olen = pattern->uncompressed_len + 5;
	if (impl->decompress(in, pattern->compressed_len, out, &olen) != 0) {
		printf("Decompression failed\n");
		return EXIT_FAILURE;
	}

	if (olen != pattern->uncompressed_len ||
	    memcmp(out, pattern->uncompressed, pattern->uncompressed_len) !=
		    0) {
		printf("Invalid decompression result\n");
		printf("Input (%zu bytes):\n", pattern->compressed_len);
		test842_hexdump(pattern->compressed, pattern->compressed_len);
		printf("Expected output (%zu bytes):\n",
		       pattern->uncompressed_len);
		test842_hexdump(pattern->uncompressed,
				pattern->uncompressed_len);
		printf("Actual output (%zu bytes):\n", olen);
		test842_hexdump(out, olen);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
