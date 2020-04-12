#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>

#if defined(USEAIX)
#include <errno.h>
#include <sys/types.h>
#include <sys/vminfo.h>
#define ALIGNMENT 4096
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

#define STRLEN 32

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

int main(int argc, const char *argv[])
{
	uint8_t *in, *out, *decompressed;
	in = out = decompressed = NULL;
	size_t ilen, olen, dlen;
	ilen = olen = dlen = 0;
	long long timestart_comp, timeend_comp;
	long long timestart_decomp, timeend_decomp;
	long long timestart_condense, timeend_condense;
#ifdef USEAIX
	int ret = 0;
#endif

	if (argc <= 1) {
		ilen = STRLEN;
		olen = ilen * 2;
#ifdef USEHW
		dlen = ilen * 2;
#else
		dlen = ilen;
#endif
		in = alloc_chunk(ilen);
		if (in == NULL) {
			printf("in = alloc_chunk(...) failed!\n");
			exit(-1);
		}
		out = alloc_chunk(olen);
		if (out == NULL) {
			printf("out = alloc_chunk(...) failed!\n");
			exit(-1);
		}
		decompressed = alloc_chunk(dlen);
		if (decompressed == NULL) {
			printf("decompressed = alloc_chunk(...) failed!\n");
			exit(-1);
		}
		uint8_t tmp[] = {
			0x30, 0x30, 0x31, 0x31, 0x32, 0x32, 0x33, 0x33,
			0x34, 0x34, 0x35, 0x35, 0x36, 0x36, 0x37, 0x37,
			0x38, 0x38, 0x39, 0x39, 0x40, 0x40, 0x41, 0x41,
			0x42, 0x42, 0x43, 0x43, 0x44, 0x44, 0x45, 0x45
		}; //"0011223344556677889900AABBCCDDEE";

		memset(in, 0, ilen);
		memset(out, 0, olen);
		memset(decompressed, 0, dlen);

		memcpy(in, tmp, STRLEN);
	} else if (argc == 2) {
		FILE *fp;
		fp = fopen(argv[1], "r");
		fseek(fp, 0, SEEK_END);
		size_t flen = (size_t)ftell(fp);
		printf("original file length: %zu\n", flen);
		ilen = nextMultipleOfChunkSize(flen);
		printf("original file length (padded): %zu\n", ilen);
		olen = ilen * 2;
#ifdef USEHW
		dlen = ilen * 2;
#else
		dlen = ilen;
#endif
		fseek(fp, 0, SEEK_SET);

		in = alloc_chunk(ilen);
		if (in == NULL) {
			printf("in = alloc_chunk(...) failed!\n");
			exit(-1);
		}
		out = alloc_chunk(olen);
		if (out == NULL) {
			printf("out = alloc_chunk(...) failed!\n");
			exit(-1);
		}
		decompressed = alloc_chunk(dlen);
		if (decompressed == NULL) {
			printf("decompressed = alloc_chunk(...) failed!\n");
			exit(-1);
		}
		memset(in, 0, ilen);
		memset(out, 0, olen);
		memset(decompressed, 0, dlen);

		if (!fread(in, flen, 1, fp)) {
			fprintf(stderr,
				"FAIL: Reading file content to memory failed.\n");
		}
		fclose(fp);
	}

	if (ilen > CHUNK_SIZE) {
		printf("Using chunks of %zu bytes\n", CHUNK_SIZE);
#ifdef USEAIX
		ret = 0;
#endif

		size_t num_chunks = ilen / CHUNK_SIZE;
		size_t *compressedChunkPositions =
			(size_t *)malloc(sizeof(size_t) * num_chunks);
		size_t *compressedChunkSizes =
			(size_t *)malloc(sizeof(size_t) * num_chunks);

		timestart_comp = timestamp();
#pragma omp parallel for
		for (size_t chunk_num = 0; chunk_num < num_chunks;
		     chunk_num++) {
			size_t chunk_olen = CHUNK_SIZE * 2;
			uint8_t *chunk_in = in + (CHUNK_SIZE * chunk_num);
			uint8_t *chunk_out =
				out + ((CHUNK_SIZE * 2) * chunk_num);

#ifdef USEAIX
			ret = accel_compress(chunk_in, CHUNK_SIZE, chunk_out,
					     &chunk_olen, 0);

			if (ret != ERANGE && ret < 0 && ret != ERANGE) {
				printf("Error calling 'accel_compress' (%d): %s\n",
				       errno, strerror(errno));
			}
			//printf("Compression factor: %f\n", (float) chunk_olen / (float) CHUNK_SIZE);
#else
			lib842_compress(chunk_in, CHUNK_SIZE, chunk_out,
					&chunk_olen);
#endif
			compressedChunkSizes[chunk_num] = chunk_olen;
		}
		timeend_comp = timestamp();

#ifndef USEAIX
		timestart_condense = timeend_comp;
#endif

		size_t currentChunkPos = 0;

		for (size_t chunk_num = 0; chunk_num < num_chunks;
		     chunk_num++) {
#ifndef USEAIX
			compressedChunkPositions[chunk_num] = currentChunkPos;
#endif
			currentChunkPos += compressedChunkSizes[chunk_num];
		}

#ifdef USEAIX
		size_t chunk_olen = CHUNK_SIZE * 2;
#else
		uint8_t *out_condensed = (uint8_t *)malloc(currentChunkPos);

#pragma omp parallel for
		for (size_t chunk_num = 0; chunk_num < num_chunks;
		     chunk_num++) {
			uint8_t *chunk_out =
				out + ((CHUNK_SIZE * 2) * chunk_num);
			uint8_t *chunk_condensed =
				out_condensed +
				compressedChunkPositions[chunk_num];
			memcpy(chunk_condensed, chunk_out,
			       compressedChunkSizes[chunk_num]);
		}
		timeend_condense = timestamp();
#endif

		timestart_decomp = timestamp();
#pragma omp parallel for
		for (size_t chunk_num = 0; chunk_num < num_chunks;
		     chunk_num++) {
			size_t chunk_dlen = CHUNK_SIZE;

			uint8_t *chunk_in = in + (CHUNK_SIZE * chunk_num);
#ifdef USEAIX
			uint8_t *chunk_out =
				out + ((CHUNK_SIZE * 2) * chunk_num);
			uint8_t *chunk_decomp = in + (CHUNK_SIZE * chunk_num);

			ret = accel_decompress(chunk_out,
					       compressedChunkSizes[chunk_num],
					       chunk_decomp, &chunk_dlen, 0);
			if (ret < 0) {
				printf("Error calling 'accel_decompress' (%d): %s\n",
				       errno, strerror(errno));
			}
#else
			uint8_t *chunk_condensed =
				out_condensed +
				compressedChunkPositions[chunk_num];
			uint8_t *chunk_decomp =
				decompressed + (CHUNK_SIZE * chunk_num);

			lib842_decompress(chunk_condensed,
					  compressedChunkSizes[chunk_num],
					  chunk_decomp, &chunk_dlen);
#endif

			if (!(memcmp(chunk_in, chunk_decomp, CHUNK_SIZE) ==
			      0)) {
				fprintf(stderr,
					"FAIL: Decompressed data differs from the original input data.\n");
				//return -1;
			}
		}
		timeend_decomp = timestamp();

#ifndef USEAIX
		free(compressedChunkPositions);
		free(compressedChunkSizes);
#endif

		printf("Input: %zu bytes\n", ilen);
		printf("Output: %zu bytes\n", currentChunkPos);
		printf("Compression factor: %f\n",
		       (float)currentChunkPos / (float)ilen);
		printf("Compression performance: %lld ms / %f MiB/s\n",
		       timeend_comp - timestart_comp,
		       (ilen / 1024 / 1024) / ((float)(timeend_comp - timestart_comp) / 1000));
#ifndef USEAIX
		printf("Condensation performance: %lld ms / %f MiB/s\n",
		       timeend_condense - timestart_condense,
		       (currentChunkPos / 1024 / 1024) / ((float)(timeend_condense - timestart_condense) / 1000));
#endif
		printf("Decompression performance: %lld ms / %f MiB/s\n",
		       timeend_decomp - timestart_decomp,
		       (ilen / 1024 / 1024) / ((float)(timeend_decomp - timestart_decomp) / 1000));

		printf("Compression- and decompression was successful!\n");
	} else {
#ifdef USEAIX
		ret = accel_compress(in, ilen, out, &olen, 0);
		if (ret < 0) {
			printf("Error calling 'accel_compress' (%d): %s\n",
			       errno, strerror(errno));
		}

		ret = accel_decompress(out, olen, decompressed, &dlen, 0);
		if (ret < 0) {
			printf("Error calling 'accel_decompress' (%d): %s\n",
			       errno, strerror(errno));
		}
#else
		lib842_compress(in, ilen, out, &olen);
		lib842_decompress(out, olen, decompressed, &dlen);
#endif

		printf("Input: %zu bytes\n", ilen);
		printf("Output: %zu bytes\n", olen);
		printf("Compression factor: %f\n", (float)olen / (float)ilen);

#ifdef USEAIX
		/*
		for (size_t i = 0; i < ilen; i++) {
			printf("%02x:", in[i]);
		}

		printf("\n\n");

		for (size_t i = 0; i < olen; i++) {
			printf("%02x:", out[i]);
		}

		printf("\n\n");
		*/
#else
		//if(argc <= 1) {
		for (int i = 0; i < 64; i++) {
			printf("%02x:", out[i]);
		}

		printf("\n\n");

		for (int i = 0; i < 32; i++) {
			printf("%02x:", decompressed[i]);
		}
		//}
#endif

		printf("\n\n");

		if (memcmp(in, decompressed, ilen) == 0) {
			printf("Compression- and decompression was successful!\n");
		} else {
			fprintf(stderr,
				"FAIL: Decompressed data differs from the original input data.\n");
			return -1;
		}
	}
}
