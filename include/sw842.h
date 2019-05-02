#ifndef __SW842_H__
#define __SW842_H__

#include <stdint.h>


int sw842_compress(const uint8_t *in, unsigned int ilen,
		   uint8_t *out, unsigned int *olen);

int sw842_decompress(const uint8_t *in, unsigned int ilen,
		     uint8_t *out, unsigned int *olen);

#endif
