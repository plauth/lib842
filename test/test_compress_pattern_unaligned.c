// Tests the case where the input and output buffers are unaligned
// (not aligned to a 8-byte boundary)
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
		printf("test_compress_pattern_unaligned IMPL PATTERN\n");
		return EXIT_FAILURE;
	}

	// Note: We overallocate recovered_in a bit (5 bytes), to make sure
	// the decompressor recovers the correct uncompressed length
	alignas(8) uint8_t inb[pattern->uncompressed_len + 3],
		outb[(pattern->uncompressed_len * 2 + 8) + 3],
		recovered_inb[pattern->uncompressed_len + 5 + 3];
	uint8_t *in = inb + 3, *out = outb + 3,
		*recovered_in = recovered_inb + 3;
	memcpy(in, pattern->uncompressed, pattern->uncompressed_len);
	size_t olen = pattern->uncompressed_len * 2 + 8,
	       recovered_ilen = pattern->uncompressed_len + 5;
	if (impl->compress(in, pattern->uncompressed_len, out, &olen) != 0) {
		printf("Compression failed\n");
		return EXIT_FAILURE;
	}
	if (impl->decompress(out, olen, recovered_in, &recovered_ilen) != 0) {
		printf("Decompression failed\n");
		return EXIT_FAILURE;
	}

	if (recovered_ilen != pattern->uncompressed_len ||
	    memcmp(recovered_in, pattern->uncompressed,
		   pattern->uncompressed_len) != 0) {
		printf("Invalid compression result\n");
		printf("Input & expected output (%zu bytes):\n",
		       pattern->uncompressed_len);
		test842_hexdump(pattern->uncompressed,
				pattern->uncompressed_len);
		printf("Actual output (%zu bytes):\n", olen);
		test842_hexdump(recovered_in, recovered_ilen);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
