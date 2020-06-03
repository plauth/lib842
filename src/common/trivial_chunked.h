#ifndef LIB842_SRC_COMMON_TRIVIAL_CHUNKED_H
#define LIB842_SRC_COMMON_TRIVIAL_CHUNKED_H

#include <stdint.h>
#include <stddef.h>

// Defines the multi-chunk versions of the compression and decompression functions
// by just calling the single-chunk versions sequentially, for cases where the
// implementation can't benefit from batching of multiple chunks
#define LIB842_DEFINE_TRIVIAL_CHUNKED_COMPRESS(chunked_compress_func, simple_compress_func) \
int chunked_compress_func(size_t numchunks, \
			  const uint8_t *in, size_t isize, const size_t *ilens, \
			  uint8_t *out, size_t osize, size_t *olens) { \
	int ret; \
	unsigned int sstride = isize / numchunks; \
	unsigned int dstride = osize / numchunks; \
 \
	for (unsigned int i = 0, soffset = 0, doffset = 0; \
	     i < numchunks; \
	     i++, soffset += sstride, doffset += dstride) { \
		ret = simple_compress_func( \
			in + soffset, ilens[i], \
			out + doffset, &olens[i]); \
		if (ret) \
			return ret; \
	} \
 \
	return 0; \
}

#define LIB842_DEFINE_TRIVIAL_CHUNKED_DECOMPRESS LIB842_DEFINE_TRIVIAL_CHUNKED_COMPRESS

#endif // LIB842_SRC_COMMON_TRIVIAL_CHUNKED_H
