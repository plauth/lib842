#ifndef LIB842_SW_H
#define LIB842_SW_H

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

#endif // LIB842_SW_H
