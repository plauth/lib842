#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <sys/time.h>

#if defined(USEAIX)
#include <sys/types.h>
#include <sys/vminfo.h>
#define ALIGNMENT 4096
#define lib842_decompress(in, ilen, out, olen) accel_decompress(in, ilen, out, olen, 0)
#define lib842_compress(in, ilen, out, olen) accel_compress(in, ilen, out, olen, 0)
#elif defined(USEHW)
#include "hw842.h"
#define lib842_decompress hw842_decompress
#define lib842_compress hw842_compress
#elif defined(USEOPTSW)
#include "sw842.h"
#define lib842_decompress optsw842_decompress
#define lib842_compress optsw842_compress
#else
#include "sw842.h"
#define lib842_decompress sw842_decompress
#define lib842_compress sw842_compress
#endif

//#define CHUNK_SIZE ((size_t)32768)
//#define CHUNK_SIZE ((size_t)1024)
#define CHUNK_SIZE ((size_t)4096)

//#define CONDENSE

static void *alloc_chunk(size_t size)
{
#ifdef ALIGNMENT
	size_t padded_size = (size + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);
	return aligned_alloc(ALIGNMENT, padded_size);
#else
	return malloc(size);
#endif
}

long long timestamp()
{
	struct timeval te;
	gettimeofday(&te, NULL);
	long long ms = te.tv_sec * 1000LL + te.tv_usec / 1000;
	return ms;
}

size_t nextMultipleOfChunkSize(size_t input)
{
	return (input + (CHUNK_SIZE - 1)) & ~(CHUNK_SIZE - 1);
}

static uint8_t *read_file(const char *file_name, size_t *ilen)
{
	FILE *fp = fopen(file_name, "rb");
	if (fp == NULL) {
		fprintf(stderr, "FAIL: Could not open the file at path '%s'.\n",
			file_name);
		return NULL;
	}

	if (fseek(fp, 0, SEEK_END) != 0) {
		fprintf(stderr, "FAIL: Could not seek the file to the end.\n");
		goto fail_file;
	}

	long flen = ftell(fp);
	if (flen == -1) {
		fprintf(stderr, "FAIL: Could not get the file length.\n");
		goto fail_file;
	}

	if (fseek(fp, 0, SEEK_SET) != 0) {
		fprintf(stderr, "FAIL: Could not seek the file to the start.\n");
		goto fail_file;
	}

	*ilen = nextMultipleOfChunkSize((size_t)flen);

	uint8_t *file_data = alloc_chunk(*ilen);
	if (file_data == NULL) {
		fprintf(stderr, "FAIL: Could not allocate memory to read the file.\n");
		goto fail_file;
	}

	memset(file_data, 0, *ilen);
	if (fread(file_data, 1, (size_t)flen, fp) != (size_t)flen) {
		fprintf(stderr,
			"FAIL: Reading file content to memory failed.\n");
		goto fail_file_data_and_file;
	}
	fclose(fp);

	printf("original file length: %li\n", flen);
	printf("original file length (padded): %zu\n", *ilen);
	return file_data;

fail_file_data_and_file:
	free(file_data);
fail_file:
	fclose(fp);
	return NULL;
}

static uint8_t *get_test_string(size_t *ilen) {
	static const uint8_t TEST_STRING[] = {
		0x30, 0x30, 0x31, 0x31, 0x32, 0x32, 0x33, 0x33,
		0x34, 0x34, 0x35, 0x35, 0x36, 0x36, 0x37, 0x37,
		0x38, 0x38, 0x39, 0x39, 0x40, 0x40, 0x41, 0x41,
		0x42, 0x42, 0x43, 0x43, 0x44, 0x44, 0x45, 0x45
	}; //"0011223344556677889900AABBCCDDEE";

	*ilen = sizeof(TEST_STRING);
	uint8_t *test_string = alloc_chunk(*ilen);
	if (test_string == NULL) {
		fprintf(stderr, "FAIL: Could not allocate memory for the test string.\n");
		return NULL;
	}

	memcpy(test_string, TEST_STRING, sizeof(TEST_STRING));
	return test_string;
}

static void compress_benchmark(const uint8_t *in, uint8_t *out, uint8_t *decompressed, size_t ilen) {
	printf("Using chunks of %zu bytes\n", CHUNK_SIZE);
	bool fail = false;

	size_t num_chunks = ilen / CHUNK_SIZE;
#ifdef CONDENSE
	size_t *compressedChunkPositions = malloc(sizeof(size_t) * num_chunks);
#endif
	size_t *compressedChunkSizes = malloc(sizeof(size_t) * num_chunks);

	long long timestart_comp = timestamp();
#pragma omp parallel for
	for (size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
		size_t chunk_olen = CHUNK_SIZE * 2;
		const uint8_t *chunk_in = in + (CHUNK_SIZE * chunk_num);
		uint8_t *chunk_out =
			out + ((CHUNK_SIZE * 2) * chunk_num);

		int ret = lib842_compress(chunk_in, CHUNK_SIZE, chunk_out,
				&chunk_olen);
		if (ret < 0) {
			printf("Error during compression (%d): %s\n",
			       errno, strerror(errno));
			#pragma omp atomic write
			fail = true;
		}
		compressedChunkSizes[chunk_num] = chunk_olen;
	}
	long long timeend_comp = timestamp();

	if (fail)
		exit(-1);

#ifdef CONDENSE
	long long timestart_condense = timeend_comp;
#endif

	size_t currentChunkPos = 0;

	for (size_t chunk_num = 0; chunk_num < num_chunks;
	     chunk_num++) {
#ifdef CONDENSE
		compressedChunkPositions[chunk_num] = currentChunkPos;
#endif
		currentChunkPos += compressedChunkSizes[chunk_num];
	}

#ifdef CONDENSE
	uint8_t *out_condensed = malloc(currentChunkPos);

#pragma omp parallel for
	for (size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
		uint8_t *chunk_out =
			out + ((CHUNK_SIZE * 2) * chunk_num);
		uint8_t *chunk_condensed =
			out_condensed +
			compressedChunkPositions[chunk_num];
		memcpy(chunk_condensed, chunk_out,
		       compressedChunkSizes[chunk_num]);
	}
	long long timeend_condense = timestamp();
#endif

	long long timestart_decomp = timestamp();
#pragma omp parallel for
	for (size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
		size_t chunk_dlen = CHUNK_SIZE;

		const uint8_t *chunk_in = in + (CHUNK_SIZE * chunk_num);
#ifdef CONDENSE
		uint8_t *chunk_out = out_condensed + compressedChunkPositions[chunk_num];
#else
		uint8_t *chunk_out = out + ((CHUNK_SIZE * 2) * chunk_num);
#endif
		uint8_t *chunk_decomp =
			decompressed + (CHUNK_SIZE * chunk_num);
		int ret = lib842_decompress(chunk_out,
				  compressedChunkSizes[chunk_num],
				  chunk_decomp, &chunk_dlen);
		if (ret < 0) {
			printf("Error during decompression (%d): %s\n",
			       errno, strerror(errno));
			#pragma omp atomic write
			fail = true;
		} else if (memcmp(chunk_in, chunk_decomp, CHUNK_SIZE) != 0) {
			fprintf(stderr,
				"FAIL: Decompressed data differs from the original input data.\n");
			#pragma omp atomic write
			fail = true;
		}
	}
	long long timeend_decomp = timestamp();

	if (fail)
		exit(-1);

#ifdef CONDENSE
	free(compressedChunkPositions);
#endif
	free(compressedChunkSizes);

	printf("Input: %zu bytes\n", ilen);
	printf("Output: %zu bytes\n", currentChunkPos);
	printf("Compression factor: %f\n",
	       (float)currentChunkPos / (float)ilen);
	printf("Compression performance: %lld ms / %f MiB/s\n",
	       timeend_comp - timestart_comp,
	       (ilen / 1024 / 1024) / ((float)(timeend_comp - timestart_comp) / 1000));
#ifdef CONDENSE
	printf("Condensation performance: %lld ms / %f MiB/s\n",
	       timeend_condense - timestart_condense,
	       (currentChunkPos / 1024 / 1024) / ((float)(timeend_condense - timestart_condense) / 1000));
#endif
	printf("Decompression performance: %lld ms / %f MiB/s\n",
	       timeend_decomp - timestart_decomp,
	       (ilen / 1024 / 1024) / ((float)(timeend_decomp - timestart_decomp) / 1000));

	printf("Compression- and decompression was successful!\n");
}


static void simple_test(const uint8_t *in, uint8_t *out, uint8_t *decompressed, size_t ilen, size_t olen, size_t dlen)
{
	int ret = 0;

	ret = lib842_compress(in, ilen, out, &olen);
	if (ret < 0) {
		printf("Error during compression (%d): %s\n",
		       errno, strerror(errno));
	}

	ret = lib842_decompress(out, olen, decompressed, &dlen);
	if (ret < 0) {
		printf("Error during decompression (%d): %s\n",
		       errno, strerror(errno));
	}

	printf("Input: %zu bytes\n", ilen);
	printf("Output: %zu bytes\n", olen);
	printf("Compression factor: %f\n", (float)olen / (float)ilen);


	for (int i = 0; i < 64; i++) {
		printf("%02x:", out[i]);
	}

	printf("\n\n");

	for (int i = 0; i < 32; i++) {
		printf("%02x:", decompressed[i]);
	}

	printf("\n\n");

	if (memcmp(in, decompressed, ilen) == 0) {
		printf("Compression- and decompression was successful!\n");
	} else {
		fprintf(stderr,
			"FAIL: Decompressed data differs from the original input data.\n");
		exit(-1);
	}
}

int main(int argc, const char *argv[])
{
	size_t ilen;
	uint8_t *in = (argc <= 1) ? get_test_string(&ilen) : read_file(argv[1], &ilen);
	if (in == NULL)
		return EXIT_FAILURE;

	size_t olen, dlen;
	olen = ilen * 2;
#ifdef USEHW
	dlen = ilen * 2;
#else
	dlen = ilen;
#endif

	uint8_t *out = alloc_chunk(olen);
	if (out == NULL) {
		printf("out = alloc_chunk(...) failed!\n");
		exit(-1);
	}
	uint8_t *decompressed = alloc_chunk(dlen);
	if (decompressed == NULL) {
		printf("decompressed = alloc_chunk(...) failed!\n");
		exit(-1);
	}
	memset(out, 0, olen);
	memset(decompressed, 0, dlen);

	if (ilen > CHUNK_SIZE) {
		compress_benchmark(in, out, decompressed, ilen);
	} else {
		simple_test(in, out, decompressed, ilen, olen, dlen);
	}

	free(in);
	free(out);
	free(decompressed);
	return EXIT_SUCCESS;
}
