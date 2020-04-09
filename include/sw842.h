#ifndef __SW842_H__
#define __SW842_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int sw842_compress(const uint8_t *in, size_t ilen,
		   uint8_t *out, size_t *olen);

int sw842_decompress(const uint8_t *in, size_t ilen,
		     uint8_t *out, size_t *olen);

int optsw842_compress(const uint8_t *in, size_t ilen,
		    uint8_t *out, size_t *olen);

int optsw842_decompress(const uint8_t *in, size_t ilen,
		      uint8_t *out, size_t *olen);

#ifdef __cplusplus
}
#endif

#endif
