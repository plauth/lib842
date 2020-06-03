#ifndef LIB842_AIX_H
#define LIB842_AIX_H

#include <lib842/config.h>
#include <lib842/common.h>

#ifdef LIB842_HAVE_AIX_HWCOMPRESSION

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int aix842_compress(const uint8_t *in, size_t ilen,
		    uint8_t *out, size_t *olen);

int aix842_decompress(const uint8_t *in, size_t ilen,
		      uint8_t *out, size_t *olen);

int aix842_compress_chunked(size_t numchunks,
			    const uint8_t *in, size_t isize, const size_t *ilens,
			    uint8_t *out, size_t osize, size_t *olens);

int aix842_decompress_chunked(size_t numchunks,
			      const uint8_t *in, size_t isize, const size_t *ilens,
			      uint8_t *out, size_t osize, size_t *olens);

const struct lib842_implementation *get_aix842_implementation();

#ifdef __cplusplus
}
#endif

#endif

#endif // LIB842_AIX_H
