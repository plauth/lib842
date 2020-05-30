#include "compdecomp_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/time.h>

long long timestamp()
{
	struct timeval te;
	gettimeofday(&te, NULL);
	long long ms = te.tv_sec * 1000LL + te.tv_usec / 1000;
	return ms;
}

void *allocate_aligned(size_t size, size_t alignment)
{
	if (alignment == 0)
		return malloc(size);

	size_t padded_size = (size + (alignment - 1)) & ~(alignment - 1);
	return aligned_alloc(alignment, padded_size);
}

static size_t nextMultipleOfChunkSize(size_t input, size_t chunk_size)
{
	return (input + (chunk_size - 1)) & ~(chunk_size - 1);
}

static uint8_t *read_file(const char *file_name, size_t *ilen, size_t chunk_size, size_t alignment)
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

	*ilen = nextMultipleOfChunkSize((size_t)flen, chunk_size);

	uint8_t *file_data = allocate_aligned(*ilen, alignment);
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

static uint8_t *get_test_string(size_t *ilen, size_t alignment) {
	static const uint8_t TEST_STRING[] = {
		0x30, 0x30, 0x31, 0x31, 0x32, 0x32, 0x33, 0x33,
		0x34, 0x34, 0x35, 0x35, 0x36, 0x36, 0x37, 0x37,
		0x38, 0x38, 0x39, 0x39, 0x40, 0x40, 0x41, 0x41,
		0x42, 0x42, 0x43, 0x43, 0x44, 0x44, 0x45, 0x45
	}; //"0011223344556677889900AABBCCDDEE";

	*ilen = sizeof(TEST_STRING);
	uint8_t *test_string = allocate_aligned(*ilen, alignment);
	if (test_string == NULL) {
		fprintf(stderr, "FAIL: Could not allocate memory for the test string.\n");
		return NULL;
	}

	memcpy(test_string, TEST_STRING, sizeof(TEST_STRING));
	return test_string;
}

static bool compress_benchmark(const uint8_t *in, size_t ilen) {
	size_t olen, dlen;
	long long time_comp, time_condense, time_decomp;
	if (!compress_benchmark_core(in, ilen, &olen, &dlen,
				     &time_comp, &time_condense, &time_decomp))
		return false;

	printf("Input: %zu bytes\n", ilen);
	printf("Output: %zu bytes\n", olen);
	printf("Compression factor: %f\n",
	       (float)olen / (float)ilen);
	if (time_comp != -1) {
		printf("Compression performance: %lld ms / %f MiB/s\n",
		       time_comp, (ilen / 1024 / 1024) / ((float)time_comp / 1000));
	}
	if (time_condense != -1) {
		printf("Condensation performance: %lld ms / %f MiB/s\n",
		       time_condense, (olen / 1024 / 1024) / ((float)time_condense / 1000));
	}
	if (time_decomp != -1) {
		printf("Decompression performance: %lld ms / %f MiB/s\n",
		       time_decomp, (dlen / 1024 / 1024) / ((float)time_decomp / 1000));
	}
	printf("Compression- and decompression was successful!\n");
	return true;
}

static bool simple_test(const uint8_t *in, size_t ilen,
			uint8_t *out, size_t olen,
			uint8_t *decompressed, size_t dlen)
{
	if (!simple_test_core(in, ilen, out, &olen, decompressed, &dlen))
		return false;

	printf("Input: %zu bytes\n", ilen);
	printf("Output: %zu bytes\n", olen);
	printf("Compression factor: %f\n", (float)olen / (float)ilen);

	for (size_t i = 0; i < olen; i++) {
		printf("%02x:", out[i]);
	}

	printf("\n\n");

	for (size_t i = 0; i < dlen; i++) {
		printf("%02x:", decompressed[i]);
	}

	printf("\n\n");

	if (ilen != dlen || memcmp(in, decompressed, ilen) != 0) {
		fprintf(stderr,
			"FAIL: Decompressed data differs from the original input data.\n");
		return false;
	}

	printf("Compression- and decompression was successful!\n");
	return true;
}

int compdecomp(const char *file_name, size_t chunk_size, size_t alignment)
{
	int ret = EXIT_FAILURE;

	size_t ilen;
	uint8_t *in = file_name != NULL
		? read_file(file_name, &ilen, chunk_size, alignment)
		: get_test_string(&ilen, alignment);
	if (in == NULL)
		return ret;

	if (ilen > chunk_size) {
		printf("Using chunks of %zu bytes\n", chunk_size);
		if (!compress_benchmark(in, ilen))
			goto return_free_in;
	} else {
		printf("Running simple test\n");

		size_t olen = ilen * 2;
		uint8_t *out = allocate_aligned(olen, alignment);
		if (out == NULL) {
			fprintf(stderr, "FAIL: out = allocate_aligned(...) failed!\n");
			goto return_free_in;
		}
		memset(out, 0, olen);

		size_t dlen = ilen;
		uint8_t *decompressed = allocate_aligned(dlen, alignment);
		if (decompressed == NULL) {
			fprintf(stderr, "FAIL: decompressed = allocate_aligned(...) failed!\n");
			goto return_free_out;
		}
		memset(decompressed, 0, dlen);

		if (!simple_test(in, ilen, out, olen, decompressed, dlen))
			goto return_free_decompressed;

return_free_decompressed:
		free(decompressed);
return_free_out:
		free(out);
		goto return_free_in;
	}

	ret = EXIT_SUCCESS;

return_free_in:
	free(in);
	return ret;
}
