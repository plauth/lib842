#include <stdio.h>
#include <stdlib.h>
#include "842.h"

#define OUT_LEN 32

int main( int argc, const char* argv[])
{
	char test[32] = "HHeelllloo  WWoorrlldd";
	uint8_t* in = (uint8_t*) test;
	unsigned int ilen = 32;
	uint8_t* out = (uint8_t*) malloc(OUT_LEN);
	memset(out, 0, OUT_LEN);
	unsigned int olen = OUT_LEN;
	void* wmem = malloc(sizeof(struct sw842_param)); 
	
	
	for (int i = 0; i < 32; i++)
	{
	    if (i > 0) printf(" ");
	    printf("%02X", test[i]);
	}
	printf("\n");
	
	sw842_compress(in, ilen, out, &olen, wmem);
	
	for (int i = 0; i < OUT_LEN; i++)
	{
	    if (i > 0) printf(" ");
	    printf("%02X", out[i]);
	}
	printf("\n");
	
}