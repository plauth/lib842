#ifndef LIB842_SW_H
#define LIB842_SW_H

#include <lib842/common.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int sw842_compress(const uint8_t *in, size_t ilen,
		   uint8_t *out, size_t *olen);

int sw842_decompress(const uint8_t *in, size_t ilen,
		     uint8_t *out, size_t *olen);

int sw842_compress_chunked(size_t numchunks,
			   const uint8_t *in, size_t isize, const size_t *ilens,
			   uint8_t *out, size_t osize, size_t *olens);

int sw842_decompress_chunked(size_t numchunks,
			     const uint8_t *in, size_t isize, const size_t *ilens,
			     uint8_t *out, size_t osize, size_t *olens);

const struct lib842_implementation *get_sw842_implementation();

int optsw842_compress(const uint8_t *in, size_t ilen,
		      uint8_t *out, size_t *olen);

int optsw842_decompress(const uint8_t *in, size_t ilen,
			uint8_t *out, size_t *olen);

int optsw842_compress_chunked(size_t numchunks,
			      const uint8_t *in, size_t isize, const size_t *ilens,
			      uint8_t *out, size_t osize, size_t *olens);

int optsw842_decompress_chunked(size_t numchunks,
			        const uint8_t *in, size_t isize, const size_t *ilens,
			        uint8_t *out, size_t osize, size_t *olens);

const struct lib842_implementation *get_optsw842_implementation();

#ifdef __cplusplus
}
#endif

#endif // LIB842_SW_H
