// Tests that decompressing an invalid bitstream fails
// This generates various pseudo-random streams to try to catch more weaknesses
// (basically, uses a primitive fuzz test)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include "test_util.h"

static unsigned xorshift_seed = 12345;
unsigned xorshift_next()
{
	xorshift_seed ^= xorshift_seed << 13;
	xorshift_seed ^= xorshift_seed >> 17;
	xorshift_seed ^= xorshift_seed << 5;
	return xorshift_seed;
}

int main(int argc, char *argv[])
{
	const struct lib842_implementation *impl;
	if (argc != 2 || (impl = test842_get_impl_by_name(argv[1])) == NULL) {
		printf("test_decompress_invalid_generator IMPL\n");
		return EXIT_FAILURE;
	}

	const size_t insize = 32, outsize = 1024;
	uint8_t *in = aligned_alloc(impl->required_alignment, insize);
	uint8_t *out = aligned_alloc(impl->required_alignment, outsize);

	for (size_t i = 0; i < 5000; i++) {
		size_t olen = outsize;
		for (size_t j = 0; j < insize; j++) {
			in[j] = (uint8_t)xorshift_next();
		}
		int ret = impl->decompress(in, insize, out, &olen);
		printf("[Run %zu] Decompression returned %d\n", i, ret);
	}

	return EXIT_SUCCESS;
}
