// Compression - decompression benchmark for non-GPU implementations,
// based on multithreading using OpenMP
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "compdecomp_driver.h"

#if defined(USEAIX)
#include <lib842/aix.h>
#define lib842impl (*get_aix842_implementation())
#elif defined(USEHW)
#include <lib842/hw.h>
#define lib842impl (*get_hw842_implementation())
#elif defined(USEOPTSW)
#include <lib842/sw.h>
#define lib842impl (*get_optsw842_implementation())
#else
#include <lib842/sw.h>
#define lib842impl (*get_sw842_implementation())
#endif

#define CHUNKS_PER_BATCH 16

//#define CHUNK_SIZE ((size_t)32768)
//#define CHUNK_SIZE ((size_t)1024)
#define CHUNK_SIZE ((size_t)4096)

//#define CONDENSE

bool compress_benchmark_core(const uint8_t *in, size_t ilen,
			     size_t *olen,
			     long long *time_comp,
			     long long *time_condense,
			     long long *time_decomp) {
	// -----
	// SETUP
	// -----
	bool ret = false;
	bool omp_success = true;

	if (CHUNKS_PER_BATCH > 1 && (CHUNK_SIZE % lib842impl.required_alignment) != 0) {
		fprintf(stderr, "FAIL: CHUNK_SIZE must be a multiple of the required 842 alignment\n");
		return ret;
	}

	uint8_t *out = aligned_alloc(lib842impl.preferred_alignment, ilen * 2);
	if (out == NULL) {
		fprintf(stderr, "FAIL: out = aligned_alloc(...) failed!\n");
		return ret;
	}
	memset(out, 0, ilen * 2);

	uint8_t *decompressed = aligned_alloc(lib842impl.preferred_alignment, ilen);
	if (decompressed == NULL) {
		fprintf(stderr, "FAIL: decompressed = aligned_alloc(...) failed!\n");
		goto exit_free_out;
	}
	memset(decompressed, 0, ilen);

	size_t num_chunks = ilen / CHUNK_SIZE;
#ifdef CONDENSE
	size_t *compressed_chunk_positions = malloc(sizeof(size_t) * num_chunks);
	if (compressed_chunk_positions == NULL) {
		fprintf(stderr, "FAIL: Could not allocate memory for the compressed chunk positions.\n");
		goto exit_free_decompressed;
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
#if CHUNKS_PER_BATCH > 1
	int chunk_rets[CHUNKS_PER_BATCH];
	size_t input_chunk_sizes[CHUNKS_PER_BATCH];
	for (size_t i = 0; i < CHUNKS_PER_BATCH; i++)
		input_chunk_sizes[i] = CHUNK_SIZE;
#endif

	long long timestart_comp = timestamp();
#pragma omp parallel for
	for (size_t chunk_num = 0; chunk_num < num_chunks; chunk_num += CHUNKS_PER_BATCH) {
		size_t batch_chunks = num_chunks - chunk_num < CHUNKS_PER_BATCH
			? num_chunks - chunk_num : CHUNKS_PER_BATCH;
		const uint8_t *chunk_in = in + (CHUNK_SIZE * chunk_num);
		uint8_t *chunk_out = out + ((CHUNK_SIZE * 2) * chunk_num);

		for (size_t i = 0; i < batch_chunks; i++)
			compressed_chunk_sizes[chunk_num + i] = CHUNK_SIZE * 2;
#if CHUNKS_PER_BATCH > 1
		int err = lib842impl.compress_chunked(
			batch_chunks, chunk_rets,
			chunk_in, CHUNK_SIZE, input_chunk_sizes,
			chunk_out, CHUNK_SIZE * 2, &compressed_chunk_sizes[chunk_num]);
#else
		int err = lib842impl.compress(chunk_in, CHUNK_SIZE, chunk_out,
					      &compressed_chunk_sizes[chunk_num]);
#endif
		if (err != 0) {
			bool is_first_failure;
#pragma omp atomic capture
			{ is_first_failure = omp_success; omp_success &= false; }
			if (is_first_failure) {
				fprintf(stderr, "FAIL: Error during compression (%d): %s\n",
					-err, strerror(-err));
			}
		}
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
	for (size_t chunk_num = 0; chunk_num < num_chunks; chunk_num += CHUNKS_PER_BATCH) {
		size_t batch_chunks = num_chunks - chunk_num < CHUNKS_PER_BATCH
			? num_chunks - chunk_num : CHUNKS_PER_BATCH;
#ifdef CONDENSE
		uint8_t *chunk_out = out_condensed + compressed_chunk_positions[chunk_num];
#else
		uint8_t *chunk_out = out + ((CHUNK_SIZE * 2) * chunk_num);
#endif
		uint8_t *chunk_decomp = decompressed + (CHUNK_SIZE * chunk_num);

		for (size_t i = 0; i < batch_chunks; i++)
			decompressed_chunk_sizes[chunk_num + i] = CHUNK_SIZE;
#if CHUNKS_PER_BATCH > 1
		int err = lib842impl.decompress_chunked(
			batch_chunks, chunk_rets,
			chunk_out, CHUNK_SIZE * 2, &compressed_chunk_sizes[chunk_num],
			chunk_decomp, CHUNK_SIZE, &decompressed_chunk_sizes[chunk_num]);
#else
		int err = lib842impl.decompress(chunk_out,
					        compressed_chunk_sizes[chunk_num],
					        chunk_decomp, &decompressed_chunk_sizes[chunk_num]);
#endif

		if (err != 0) {
			bool is_first_failure;
#pragma omp atomic capture
			{ is_first_failure = omp_success; omp_success &= false; }
			if (is_first_failure) {
				fprintf(stderr, "FAIL: Error during decompression (%d): %s\n",
					-err, strerror(-err));
			}
		}
	}

	if (!omp_success)
		goto exit_free_out_condensed;

	*time_decomp = timestamp() - timestart_decomp;

	size_t dlen = 0;
	for (size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++)
		dlen += decompressed_chunk_sizes[chunk_num];

	// ----------
	// VALIDATION
	// ----------
	if (ilen != dlen || memcmp(in, decompressed, ilen) != 0) {
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
exit_free_decompressed:
	free(decompressed);
exit_free_out:
	free(out);

	return ret;
}

bool simple_test_core(const uint8_t *in, size_t ilen,
		      uint8_t *out, size_t *olen,
		      uint8_t *decompressed, size_t *dlen)
{
	int err;

	err = lib842impl.compress(in, ilen, out, olen);
	if (err != 0) {
		fprintf(stderr, "Error during compression (%d): %s\n",
			-err, strerror(-err));
		return false;
	}

	err = lib842impl.decompress(out, *olen, decompressed, dlen);
	if (err != 0) {
		fprintf(stderr, "Error during decompression (%d): %s\n",
			-err, strerror(-err));
		return false;
	}

	return true;
}

int main(int argc, const char *argv[])
{
	return compdecomp(argc > 1 ? argv[1] : NULL, CHUNK_SIZE, lib842impl.preferred_alignment);
}
