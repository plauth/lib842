// Compression - decompression benchmark for non-GPU implementations,
// based on multithreading using OpenMP
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "compdecomp_driver.h"

#if defined(USEAIX)
#include <sys/types.h>
#include <sys/vminfo.h>
#define ALIGNMENT 4096
#define lib842_decompress(in, ilen, out, olen) accel_decompress(in, ilen, out, olen, 0)
#define lib842_compress(in, ilen, out, olen) accel_compress(in, ilen, out, olen, 0)
#elif defined(USEHW)
#include <lib842/hw.h>
#define ALIGNMENT 0
#define lib842_decompress hw842_decompress
#define lib842_compress hw842_compress
#elif defined(USEOPTSW)
#include <lib842/sw.h>
#define ALIGNMENT 0
#define lib842_decompress optsw842_decompress
#define lib842_compress optsw842_compress
#else
#include <lib842/sw.h>
#define ALIGNMENT 0
#define lib842_decompress sw842_decompress
#define lib842_compress sw842_compress
#endif

//#define CHUNK_SIZE ((size_t)32768)
//#define CHUNK_SIZE ((size_t)1024)
#define CHUNK_SIZE ((size_t)4096)

//#define CONDENSE

bool compress_benchmark_core(const uint8_t *in, size_t ilen,
			     uint8_t *out, size_t *olen,
			     uint8_t *decompressed, size_t *dlen,
			     long long *time_comp,
			     long long *time_condense,
			     long long *time_decomp) {
	// -----
	// SETUP
	// -----
	bool ret = false;
	bool omp_success = true;

	size_t num_chunks = ilen / CHUNK_SIZE;
#ifdef CONDENSE
	size_t *compressed_chunk_positions = malloc(sizeof(size_t) * num_chunks);
	if (compressed_chunk_positions == NULL) {
		fprintf(stderr, "FAIL: Could not allocate memory for the compressed chunk positions.\n");
		return ret;
	}
#endif
	size_t *compressed_chunk_sizes = malloc(sizeof(size_t) * num_chunks);
	if (compressed_chunk_sizes == NULL) {
		fprintf(stderr, "FAIL: Could not allocate memory for the compressed chunk sizes.\n");
		goto exit_free_compressed_chunk_positions;
	}
	size_t *decompressed_chunk_sizes = malloc(sizeof(size_t) * num_chunks);
	if (decompressed_chunk_sizes == NULL) {
		fprintf(stderr, "FAIL: Could not allocate memory for the decompressed chunk sizes.\n");
		goto exit_free_compressed_chunk_sizes;
	}

	// -----------
	// COMPRESSION
	// -----------
	long long timestart_comp = timestamp();
#pragma omp parallel for
	for (size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
		size_t chunk_olen = CHUNK_SIZE * 2;
		const uint8_t *chunk_in = in + (CHUNK_SIZE * chunk_num);
		uint8_t *chunk_out = out + ((CHUNK_SIZE * 2) * chunk_num);

		int err = lib842_compress(chunk_in, CHUNK_SIZE, chunk_out,
					  &chunk_olen);
		if (err != 0) {
			bool is_first_failure;
#pragma omp atomic capture
			{ is_first_failure = omp_success; omp_success &= false; }
			if (is_first_failure) {
				fprintf(stderr, "FAIL: Error during compression (%d): %s\n",
					-err, strerror(-err));
			}
		}
		compressed_chunk_sizes[chunk_num] = chunk_olen;
	}

	if (!omp_success)
		goto exit_free_decompressed_chunk_sizes;

	*time_comp = timestamp() - timestart_comp;

	*olen = 0;
	for (size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++)
		*olen += compressed_chunk_sizes[chunk_num];

	// ------------
	// CONDENSATION
	// ------------
#ifdef CONDENSE
	long long timestart_condense = timestamp();

	for (size_t chunk_num = 0, pos = 0; chunk_num < num_chunks; chunk_num++) {
		compressed_chunk_positions[chunk_num] = pos;
		pos += compressed_chunk_sizes[chunk_num];
	}

	uint8_t *out_condensed = malloc(*olen);
	if (out_condensed == NULL) {
		fprintf(stderr, "FAIL: Could not allocate memory for the condensed data.\n");
		goto exit_free_decompressed_chunk_sizes;
	}

#pragma omp parallel for
	for (size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
		uint8_t *chunk_out = out + ((CHUNK_SIZE * 2) * chunk_num);
		uint8_t *chunk_condensed =
			out_condensed + compressed_chunk_positions[chunk_num];
		memcpy(chunk_condensed, chunk_out,
		       compressed_chunk_sizes[chunk_num]);
	}
	*time_condense = timestamp() - timestart_condense;
#else
	*time_condense = -1;
#endif

	// -------------
	// DECOMPRESSION
	// -------------
	long long timestart_decomp = timestamp();
#pragma omp parallel for
	for (size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
		size_t chunk_dlen = CHUNK_SIZE;
#ifdef CONDENSE
		uint8_t *chunk_out = out_condensed + compressed_chunk_positions[chunk_num];
#else
		uint8_t *chunk_out = out + ((CHUNK_SIZE * 2) * chunk_num);
#endif
		uint8_t *chunk_decomp = decompressed + (CHUNK_SIZE * chunk_num);

		int err = lib842_decompress(chunk_out,
					    compressed_chunk_sizes[chunk_num],
					    chunk_decomp, &chunk_dlen);
		if (err != 0) {
			bool is_first_failure;
#pragma omp atomic capture
			{ is_first_failure = omp_success; omp_success &= false; }
			if (is_first_failure) {
				fprintf(stderr, "FAIL: Error during decompression (%d): %s\n",
					-err, strerror(-err));
			}
		}
		decompressed_chunk_sizes[chunk_num] = chunk_dlen;
	}

	if (!omp_success)
		goto exit_free_out_condensed;

	*time_decomp = timestamp() - timestart_decomp;

	*dlen = 0;
	for (size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++)
		*dlen += decompressed_chunk_sizes[chunk_num];

	// ----------
	// VALIDATION
	// ----------
	if (ilen != *dlen || memcmp(in, decompressed, ilen) != 0) {
		fprintf(stderr,
			"FAIL: Decompressed data differs from the original input data.\n");
		goto exit_free_out_condensed;
	}

	ret = true;

exit_free_out_condensed:
#ifdef CONDENSE
	free(out_condensed);
#endif
exit_free_decompressed_chunk_sizes:
	free(decompressed_chunk_sizes);
exit_free_compressed_chunk_sizes:
	free(compressed_chunk_sizes);
exit_free_compressed_chunk_positions:
#ifdef CONDENSE
	free(compressed_chunk_positions);
#endif

	return ret;
}

bool simple_test_core(const uint8_t *in, size_t ilen,
		      uint8_t *out, size_t *olen,
		      uint8_t *decompressed, size_t *dlen)
{
	int err;

	err = lib842_compress(in, ilen, out, olen);
	if (err != 0) {
		fprintf(stderr, "Error during compression (%d): %s\n",
			-err, strerror(-err));
		return false;
	}

	err = lib842_decompress(out, *olen, decompressed, dlen);
	if (err != 0) {
		fprintf(stderr, "Error during decompression (%d): %s\n",
			-err, strerror(-err));
		return false;
	}

	return true;
}

int main(int argc, const char *argv[])
{
	return compdecomp(argc > 1 ? argv[1] : NULL, CHUNK_SIZE, ALIGNMENT);
}
