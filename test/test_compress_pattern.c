// Tests that the result of compressing some data and uncompressing it
// returns back the same data.
// Compression and decompression can be done with different implementations,
// which allows validating that they are mutually compatible
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdalign.h>
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
		out[pattern->uncompressed_len * 2 + 8],
		recovered_in[pattern->uncompressed_len];
	memcpy(in, pattern->uncompressed, pattern->uncompressed_len);
	size_t olen = pattern->uncompressed_len * 2 + 8,
	       recovered_ilen = pattern->uncompressed_len;
	if (impl_compression->compress(in, pattern->uncompressed_len, out,
				       &olen) != 0) {
		printf("Compression failed\n");
		return EXIT_FAILURE;
	}
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
