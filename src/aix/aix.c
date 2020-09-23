#include "../common/trivial_chunked.h"
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

const struct lib842_implementation *get_aix842_implementation() {
	static struct lib842_implementation aix842_implementation = {
		.compress = aix842_compress,
		.decompress = aix842_decompress,
		.compress_chunked = aix842_compress_chunked,
		.decompress_chunked = aix842_decompress_chunked,
		// From https://www.ibm.com/support/knowledgecenter/ssw_aix_71/a_bostechref/accel_compress.html
		.required_alignment = 4096,
		.preferred_alignment = 128
	};
	return &aix842_implementation;
};
