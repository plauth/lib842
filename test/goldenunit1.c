#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hw842.h"
#include "sw842.h"

#define CHUNK_SIZE ((size_t)4096)


size_t nextMultipleOfChunkSize(size_t input) {
	return (input + (CHUNK_SIZE-1)) & ~(CHUNK_SIZE-1);
} 

int main( int argc, const char* argv[])
{
	uint8_t *in, *out, *decompressed;
	in = out = decompressed = NULL;
	size_t ilen, olen, dlen;
	ilen = olen = dlen = 0;

	if(argc <= 1) {
		ilen = 8;
		olen = ilen * 2;
		dlen = ilen * 2;
		in = (uint8_t*) malloc(ilen);
		out = (uint8_t*) malloc(olen);
		decompressed = (uint8_t*) malloc(dlen);

		uint8_t tmp[] = "00112233";

		memset(in, 0, ilen);
		memset(out, 0, olen);
		memset(decompressed, 0, dlen);

		memcpy(in, tmp, 8);

	} else if (argc == 2) {
		FILE *fp;
		fp=fopen(argv[1], "r");
		fseek(fp, 0, SEEK_END);
		size_t flen = (size_t)ftell(fp);
		ilen = flen;
		printf("original file length: %d\n", ilen);
		ilen = nextMultipleOfChunkSize(ilen);
		printf("original file length (padded): %d\n", ilen);
		olen = ilen * 2;
		dlen = ilen * 2;
		fseek(fp, 0, SEEK_SET);

		in = (uint8_t*) malloc(ilen);
		out = (uint8_t*) malloc(olen);
		decompressed = (uint8_t*) malloc(dlen);
		memset(in, 0, ilen);
		memset(out, 0, olen);
		memset(decompressed, 0, dlen);

		if(!fread(in, flen, 1, fp)) {
			fprintf(stderr, "FAIL: Reading file content to memory failed.\n");
		}
		fclose(fp);
	}

	if(ilen > CHUNK_SIZE) {
		printf("Using chunks of %zu bytes\n", CHUNK_SIZE);
		size_t acc_olen = 0;
		size_t chunk_dlen;

		size_t num_chunks = ilen / CHUNK_SIZE;

		for(size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
			
			size_t chunk_olen = CHUNK_SIZE * 2;
			uint8_t* chunk_in = in + (CHUNK_SIZE * chunk_num);
			uint8_t* chunk_out = out + ((CHUNK_SIZE * 2) * chunk_num);
			
			sw842_compress(chunk_in, CHUNK_SIZE, chunk_out, &chunk_olen);
			acc_olen += chunk_olen;
		}

		uint8_t* chunk_in = in;
		uint8_t* chunk_out = out;
		uint8_t* chunk_decomp = decompressed;
		size_t chunk_olen = CHUNK_SIZE * 2;

		for(size_t out_chunk_pos = 0; out_chunk_pos < olen; out_chunk_pos+=(CHUNK_SIZE * 2)) {
			chunk_dlen = CHUNK_SIZE;
			
			hw842_decompress(chunk_out, chunk_olen, chunk_decomp, &chunk_dlen);

			if (!(memcmp(chunk_in, chunk_decomp, CHUNK_SIZE) == 0)) {
				fprintf(stderr, "FAIL: Decompressed data differs from the original input data.\n");
				return -1;
			}

			chunk_in += CHUNK_SIZE;
			chunk_out += (CHUNK_SIZE*2);
			chunk_decomp += CHUNK_SIZE;
		}

		printf("Compression (SW) and decompression (HW) was successful!\n");
	} else {

		sw842_compress(in, ilen, out, &olen);
		hw842_decompress(out, olen, decompressed, &dlen);

		if (memcmp(in, decompressed, ilen) == 0) {
			printf("Compression (SW) and decompression (HW) was successful!\n");
		} else {
			fprintf(stderr, "FAIL: Decompressed data differs from the original input data.\n");
			return -1;
		}

	}
	
}
