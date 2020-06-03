// Tests the case where during decompression, the output buffer is too small
// (minus one byte, so just barely smaller than required)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include "test_patterns.h"
#include "test_util.h"

int main(int argc, char *argv[])
{
	const struct lib842_implementation *impl;
	const struct test842_pattern *pattern;
	if (argc != 3 || (impl = test842_get_impl_by_name(argv[1])) == NULL ||
	    (pattern = test842_get_pattern_by_name(argv[2])) == NULL) {
		printf("test_decompress_pattern_barelysmall IMPL PATTERN\n");
		return EXIT_FAILURE;
	}

	assert(pattern->uncompressed_len > 1);
	uint8_t *in = aligned_alloc(impl->required_alignment, pattern->compressed_len);
	memcpy(in, pattern->compressed, pattern->compressed_len);
	uint8_t *out = aligned_alloc(impl->required_alignment, pattern->uncompressed_len - 1);
	size_t olen = pattern->uncompressed_len - 1;
	if (impl->decompress(in, pattern->compressed_len, out, &olen) != -ENOSPC) {
		printf("Decompression should have failed with ENOSPC\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
