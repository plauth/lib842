// Tests that the result of compressing some data and uncompressing it
// returns back the same data.
// Compression and decompression can be done with different implementations,
// which allows validating that they are mutually compatible
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdalign.h>
#include <errno.h>
#include "test_patterns.h"
#include "test_util.h"

int main(int argc, char *argv[])
{
	const struct test842_impl *impl_compression, *impl_decompression;
	const struct test842_pattern *pattern;
	if (argc != 4 ||
	    (impl_compression = test842_get_impl_by_name(argv[1])) == NULL ||
	    (impl_decompression = test842_get_impl_by_name(argv[2])) == NULL ||
	    (pattern = test842_get_pattern_by_name(argv[3])) == NULL) {
		printf("test_compress_pattern IMPL_COMPRESSION IMPL_DECOMPRESSION PATTERN\n");
		return EXIT_FAILURE;
	}

	alignas(8) uint8_t in[pattern->uncompressed_len],
		out[pattern->uncompressed_len * 2 + 8];
	memcpy(in, pattern->uncompressed, pattern->uncompressed_len);
	size_t olen = pattern->uncompressed_len * 2 + 8;
	int err = impl_compression->compress(in, pattern->uncompressed_len, out, &olen);
	// FIXME TESTFAILURE: Compression (but not decompression) fails on real hardware
	//                    (cryptodev + nx-842) because the driver doesn't accept ilen=0
	if (err == -EINVAL && strcmp(argv[1], "hw") == 0) {
		fprintf(stderr, "!! TEST FAILED (BUT PASS, DUE TO KNOWN DEFECT) !!\n");
		return EXIT_SUCCESS;
	}
	if (err != 0) {
		printf("Compression failed\n");
		return EXIT_FAILURE;
	}
	// Note: We overallocate recovered_in a bit (5 bytes), to make sure
	// the decompressor recovers the correct uncompressed length
	// This also makes the test work on real HW (the nx-842 kernel driver
	// doesn't accept an output buffer of size 0 even if it's sufficient)
	alignas(8) uint8_t recovered_in[pattern->uncompressed_len + 5];
	size_t recovered_ilen = pattern->uncompressed_len + 5;
	if (impl_decompression->decompress(out, olen, recovered_in,
					   &recovered_ilen) != 0) {
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
