#ifndef __SW842_H__
#define __SW842_H__

#include <stdint.h>


int sw842_compress(const uint8_t *in, size_t ilen,
		   uint8_t *out, size_t *olen);

int sw842_decompress(const uint8_t *in, size_t ilen,
		     uint8_t *out, size_t *olen);

#endif
