// Tests compressing and decompressing generated data that starts as a
// random pattern, but progresses to a more stable pattern
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "test_util.h"

#define BUFFER_SIZE 2048

static unsigned xorshift_seed = 1234;
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
	if (argc != 2 ||
	    (impl = test842_get_impl_by_name(argv[1])) == NULL) {
		printf("test_compress_diffuse IMPL\n");
		return EXIT_FAILURE;
	}

	uint8_t *data_buffer = aligned_alloc(impl->required_alignment, BUFFER_SIZE),
		*tmpbuf = malloc(BUFFER_SIZE),
		*comp_data = aligned_alloc(impl->required_alignment, 2 * BUFFER_SIZE),
		*uncomp_data = aligned_alloc(impl->required_alignment, BUFFER_SIZE);
	int ret = EXIT_SUCCESS;

	// Initialize data_buffer with a random pattern
	for (size_t i = 0; i < BUFFER_SIZE; i++) {
		data_buffer[i] = (uint8_t)xorshift_next();
	}

	for (int it = 0; it < 13; it++) {
		// "Diffuse" each value in data_buffer with its neighbors.
		// This progressively increases the compressibility of the data,
		// and eventually converges to all zeroes
		for (size_t i = 0; i < BUFFER_SIZE; i++) {
			tmpbuf[i] = (uint8_t)
				((data_buffer[(i + BUFFER_SIZE - 1) % BUFFER_SIZE] +
				  data_buffer[(i + 1) % BUFFER_SIZE]) / 3);
		}
		memcpy(data_buffer, tmpbuf, BUFFER_SIZE);

		// Try to compress the data, then uncompress it, and check the result is correct
		size_t comp_size = 2 * BUFFER_SIZE;
		int ret_compress = impl->compress(data_buffer, BUFFER_SIZE,
						  comp_data, &comp_size);
		if (ret_compress != 0) {
			printf("it=%i: Compression FAILED (ret=%d)\n", it, ret_compress);
			return EXIT_FAILURE;
		}

		size_t uncomp_size = BUFFER_SIZE;
		int ret_decompress = impl->decompress(comp_data, comp_size,
						      uncomp_data, &uncomp_size);
		if (ret_decompress != 0) {
			printf("it=%i: Pattern compressed to %zu bytes but decompression FAILED (ret=%d)\n",
			       it, comp_size, ret_decompress);
			return EXIT_FAILURE;
		}

		if (memcmp(data_buffer, uncomp_data, BUFFER_SIZE) != 0) {
			printf("it=%i: Pattern compressed to %zu bytes but decompression INCORRECT\n",
			       it, comp_size);
			return EXIT_FAILURE;
		}

		printf("it=%i: Pattern compressed to %zu bytes and decompressed OK\n",
		       it, comp_size);
	}

	return ret;
}
