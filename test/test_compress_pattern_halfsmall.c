// Tests the case where during compression, the output buffer is too small
// (half length, so noticeably smaller than required)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "test_patterns.h"
#include "test_util.h"

int main(int argc, char *argv[])
{
	const struct lib842_implementation *impl;
	const struct test842_pattern *pattern;
	if (argc != 3 || (impl = test842_get_impl_by_name(argv[1])) == NULL ||
	    (pattern = test842_get_pattern_by_name(argv[2])) == NULL) {
		printf("test_compress_pattern_halfsmall IMPL PATTERN\n");
		return EXIT_FAILURE;
	}

	uint8_t *in = aligned_alloc(impl->required_alignment, pattern->uncompressed_len);
	memcpy(in, pattern->uncompressed, pattern->uncompressed_len);
	uint8_t *out = aligned_alloc(impl->required_alignment, pattern->uncompressed_len * 2 + 8);
	size_t olen = pattern->uncompressed_len * 2 + 8;
	int err = impl->compress(in, pattern->uncompressed_len, out, &olen);
	// FIXME TESTFAILURE: Compression (but not decompression) fails on real hardware
	//                    (cryptodev + nx-842) because the driver doesn't accept ilen=0
	if (err == -EINVAL && strcmp(argv[1], "hw") == 0) {
		fprintf(stderr, "!! TEST FAILED (BUT PASS, DUE TO KNOWN DEFECT) !!\n");
		return EXIT_SUCCESS;
	}
	if (err != 0) {
		printf("Setup compression failed when it shouldn't\n");
		return EXIT_FAILURE;
	}
	olen--; // Now it shouldn't fit
	if (impl->compress(in, pattern->uncompressed_len, out, &olen) != -ENOSPC) {
		printf("Compression should have failed with ENOSPC\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
