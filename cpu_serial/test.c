#include <stdio.h>
#include <stdlib.h>
#include "842.h"



int nextMultipleOfEight(unsigned int input) {
	return (input + 7) & ~7;
} 

int main( int argc, const char* argv[])
{

	void* wmem_comp = malloc(sizeof(struct sw842_param)); 
	uint8_t *in, *out, *decompressed;
	unsigned int ilen, olen, dlen;

	if(argc <= 1) {
		ilen = 8;
		olen = ilen * 2;
		dlen = ilen;
		in = (uint8_t*) malloc(ilen);
		out = (uint8_t*) malloc(olen);
		decompressed = (uint8_t*) malloc(dlen);

		uint8_t tmp[] = "00112233";

		memset(in, 0, ilen);
		memset(out, 0, olen);
		memset(decompressed, 0, dlen);

		strncpy((char *) in, (const char *) tmp, 8);

	} else if (argc == 2) {
		FILE *fp;
		fp=fopen(argv[1], "r");
		fseek(fp, 0, SEEK_END);
		unsigned int flen = ftell(fp);
		ilen = flen;
		printf("original file length: %d\n", ilen);
		ilen = nextMultipleOfEight(ilen);
		printf("original file length (padded): %d\n", ilen);
		olen = ilen * 2;
		dlen = ilen;
		fseek(fp, 0, SEEK_SET);

		in = (uint8_t*) malloc(ilen);
		out = (uint8_t*) malloc(olen);
		decompressed = (uint8_t*) malloc(dlen);
		fread(out, flen, 1, fp);
		fclose(fp);
	}

	/*
	for (int i = 0; i < 8; i++)
	{
	    if (i > 0) printf(" ");
	    printf("%02X", in[i]);
	}
	printf("\n");*/

	printf("commencing Compression\n");
	
	sw842_compress(in, ilen, out, &olen, wmem_comp);

	printf("In: %d bytes\n", ilen);
	printf("Out: %d bytes\n", olen);
	//printf("Compression factor: %f\n", (float) olen / (float) ilen);
	
	/*
	for (int i = 0; i < OUT_LEN; i++)
	{
	    if (i > 0) printf(" ");
	    printf("%02X", out[i]);
	}
	printf("\n");*/

	sw842_decompress(out, olen, decompressed, &dlen);

	/*
	for (int i = 0; i < 8; i++)
	{
	    if (i > 0) printf(" ");
	    printf("%02X", decompressed[i]);
	}
	printf("\n");*/

	if (memcmp(in, decompressed, 8) == 0) {
		printf("Compression- and decompression was successful!\n");
	} else {
		fprintf(stderr, "FAIL: Decompressed data differs from the original input data.\n");
		return -1;
	}
	
}
