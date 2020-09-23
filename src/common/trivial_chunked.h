#ifndef LIB842_SRC_COMMON_TRIVIAL_CHUNKED_H
#define LIB842_SRC_COMMON_TRIVIAL_CHUNKED_H

#include <stdint.h>
#include <stddef.h>
#include <limits.h>

// Defines the multi-chunk versions of the compression and decompression functions
// by just calling the single-chunk versions sequentially, for cases where the
// implementation can't benefit from batching of multiple chunks
#define LIB842_DEFINE_TRIVIAL_CHUNKED_COMPRESS(chunked_compress_func, simple_compress_func) \
int chunked_compress_func(size_t numchunks, int *rets, \
			  const uint8_t *in, size_t istride, const size_t *ilens, \
			  uint8_t *out, size_t ostride, size_t *olens) { \
	int ret = 0; \
	for (unsigned int i = 0, soffset = 0, doffset = 0; \
	     i < numchunks; \
	     i++, soffset += istride, doffset += ostride) { \
		if (ilens[i] == SIZE_MAX) \
			continue; \
		rets[i] = simple_compress_func( \
			in + soffset, ilens[i], \
			out + doffset, &olens[i]); \
		if (ret == 0 && (rets[i] != 0 && rets[i] != -ENOSPC)) \
			ret = rets[i]; \
	} \
 \
	return 0; \
}

#define LIB842_DEFINE_TRIVIAL_CHUNKED_DECOMPRESS LIB842_DEFINE_TRIVIAL_CHUNKED_COMPRESS

#endif // LIB842_SRC_COMMON_TRIVIAL_CHUNKED_H
