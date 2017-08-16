#include <stdio.h>
#include <stdlib.h>
#include "842.h"

#define OUT_LEN 16

int main( int argc, const char* argv[])
{
	void* wmem_comp = malloc(sizeof(struct sw842_param)); 
	uint8_t in[8];
	uint8_t out[OUT_LEN];
	uint8_t decompressed [8];
	uint8_t tmp[] = "00112233";

	unsigned int ilen = 8;
	unsigned int olen = OUT_LEN;
	unsigned int dlen = 8;

	memset(in, 0, 8);
	memset(out, 0, OUT_LEN);
	memset(decompressed, 0, OUT_LEN);

	strncpy((char *) in, (const char *) tmp, 8);

	for (int i = 0; i < 8; i++)
	{
	    if (i > 0) printf(" ");
	    printf("%02X", in[i]);
	}
	printf("\n");
	
	sw842_compress(in, ilen, out, &olen, wmem_comp);
	
	for (int i = 0; i < OUT_LEN; i++)
	{
	    if (i > 0) printf(" ");
	    printf("%02X", out[i]);
	}
	printf("\n");

	sw842_decompress(out, olen, decompressed, &dlen);

		for (int i = 0; i < 8; i++)
	{
	    if (i > 0) printf(" ");
	    printf("%02X", decompressed[i]);
	}
	printf("\n");

	if (memcmp(in, decompressed, 8) == 0) {
		printf("Compression- and decompression was successful!\n");
	} else {
		fprintf(stderr, "FAIL: Decompressed data differs from the original input data.\n");
		return -1;
	}
	
}
