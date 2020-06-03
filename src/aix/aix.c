#include <lib842/aix.h>

#include <sys/types.h>
#include <sys/vminfo.h>

int aix842_compress(const uint8_t *in, size_t ilen,
		    uint8_t *out, size_t *olen) {
	return accel_compress((uint8_t *)in, ilen, out, olen, 0);
}

int aix842_decompress(const uint8_t *in, size_t ilen,
		      uint8_t *out, size_t *olen) {
	return accel_decompress((uint8_t *)in, ilen, out, olen, 0);
}

LIB842_DEFINE_TRIVIAL_CHUNKED_COMPRESS(aix842_decompress_chunked, aix842_decompress)
LIB842_DEFINE_TRIVIAL_CHUNKED_DECOMPRESS(aix842_compress_chunked, aix842_compress)
struct lib842_implementation aix842_implementation = {
	aix842_compress,
	aix842_decompress,
	aix842_compress_chunked,
	aix842_decompress_chunked,
	4096
};
