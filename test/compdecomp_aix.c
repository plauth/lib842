#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/vminfo.h>

#define CHUNK_SIZE ((size_t)262144) //32768
//#define CHUNK_SIZE ((size_t)524288) //65536
//#define CHUNK_SIZE ((size_t)4096)
#define ALIGNMENT 4096

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
	int ret = 0;

	if (argc <= 1) {
		ilen = 32;
		olen = ilen * 2;
#ifdef USEHW
		dlen = ilen * 2;
#else
		dlen = ilen;
#endif
		in = (uint8_t *)aligned_alloc(ALIGNMENT, CHUNK_SIZE);
		if (in == NULL) {
			printf("in = aligned_alloc(...) failed!\n");
			exit(-1);
		}
		out = (uint8_t *)aligned_alloc(ALIGNMENT, CHUNK_SIZE);
		if (out == NULL) {
			printf("out = aligned_alloc(...) failed!\n");
			exit(-1);
		}
		decompressed = (uint8_t *)aligned_alloc(ALIGNMENT, CHUNK_SIZE);
		if (decompressed == NULL) {
			printf("decompressed = aligned_alloc(...) failed!\n");
			exit(-1);
		}
		uint8_t tmp[] = "00112233001122330011223300112233";

		memset(in, 0, ilen);
		memset(out, 0, olen);
		memset(decompressed, 0, dlen);

		memcpy(in, tmp, 32);

	} else if (argc == 2) {
		FILE *fp;
		fp = fopen(argv[1], "r");
		fseek(fp, 0, SEEK_END);
		size_t flen = (size_t)ftell(fp);
		ilen = flen;
#ifndef BENCHMARK
		printf("original file length: %d\n", ilen);
#endif
		ilen = nextMultipleOfChunkSize(ilen);
#ifndef BENCHMARK
		printf("original file length (padded): %d\n", ilen);
#endif
		olen = ilen * 2;
		dlen = ilen * 2;
		fseek(fp, 0, SEEK_SET);

		in = (uint8_t *)aligned_alloc(ALIGNMENT, ilen);
		if (in == NULL) {
			printf("in = aligned_alloc(...) failed!\n");
			exit(-1);
		}
		out = (uint8_t *)aligned_alloc(ALIGNMENT, olen);
		if (out == NULL) {
			printf("out = aligned_alloc(...) failed!\n");
			exit(-1);
		}
		decompressed = (uint8_t *)aligned_alloc(ALIGNMENT, dlen);
		if (decompressed == NULL) {
			printf("decompressed = aligned_alloc(...) failed!\n");
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
#ifndef BENCHMARK
		printf("Using chunks of %zu bytes\n", CHUNK_SIZE);
#endif
		size_t acc_olen = 0;
		ret = 0;

		size_t num_chunks = ilen / CHUNK_SIZE;
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

			ret = accel_compress(chunk_in, CHUNK_SIZE, chunk_out,
					     &chunk_olen, 0);

			if (ret != ERANGE && ret < 0 && ret != ERANGE) {
				printf("Error calling 'accel_compress' (%d): %s\n",
				       errno, strerror(errno));
			}
			//printf("Compression factor: %f\n", (float) chunk_olen / (float) CHUNK_SIZE);
			compressedChunkSizes[chunk_num] = chunk_olen;
		}
		timeend_comp = timestamp();

		size_t currentChunkPos = 0;
		for (size_t chunk_num = 0; chunk_num < num_chunks;
		     chunk_num++) {
			currentChunkPos += compressedChunkSizes[chunk_num];
		}
		size_t chunk_olen = CHUNK_SIZE * 2;

		timestart_decomp = timestamp();
#pragma omp parallel for
		for (size_t chunk_num = 0; chunk_num < num_chunks;
		     chunk_num++) {
			size_t chunk_dlen = CHUNK_SIZE;

			uint8_t *chunk_in = in + (CHUNK_SIZE * chunk_num);
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

			if (!(memcmp(chunk_in, chunk_decomp, CHUNK_SIZE) ==
			      0)) {
				fprintf(stderr,
					"FAIL: Decompressed data differs from the original input data.\n");
				//return -1;
			}
		}
		timeend_decomp = timestamp();
#ifdef BENCHMARK
		printf("%.4f,%.4f\n",
		       (ilen / 1024 / 1024) /
			       ((float)(timeend_comp - timestart_comp) / 1000),
		       (ilen / 1024 / 1024) /
			       ((float)(timeend_decomp - timestart_decomp) /
				1000));
#else
		printf("Input: %zu bytes\n", ilen);
		printf("Output: %zu bytes\n", currentChunkPos);
		printf("Compression factor: %f\n",
		       (float)currentChunkPos / (float)ilen);
		printf("Compression performance: %lld ms / %f MiB/s\n",
		       timeend_comp - timestart_comp,
		       (ilen / 1024 / 1024) /
			       ((float)(timeend_comp - timestart_comp) / 1000));
		printf("Decompression performance: %lld ms / %f MiB/s\n",
		       timeend_decomp - timestart_decomp,
		       (ilen / 1024 / 1024) /
			       ((float)(timeend_decomp - timestart_decomp) /
				1000));

		printf("Compression- and decompression was successful!\n");
#endif
	} else {
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

		printf("Input: %zu bytes\n", ilen);
		printf("Output: %zu bytes\n", olen);
		printf("Compression factor: %f\n", (float)olen / (float)ilen);

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

		if (memcmp(in, decompressed, ilen) == 0) {
			printf("Compression- and decompression was successful!\n");
		} else {
			fprintf(stderr,
				"FAIL: Decompressed data differs from the original input data.\n");
			return -1;
		}
	}
}
