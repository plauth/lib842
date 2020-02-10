#ifndef __HW842_H__
#define __HW842_H__

#include <stdint.h>

int hw842_compress(const uint8_t *in, size_t ilen,
		   uint8_t *out, size_t *olen);

int hw842_decompress(const uint8_t *in, size_t ilen,
		     uint8_t *out, size_t *olen);

#endif
