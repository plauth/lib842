#ifndef LIB842_HW_H
#define LIB842_HW_H

#include <lib842/config.h>
#include <lib842/common.h>

#ifdef LIB842_HAVE_CRYPTODEV_LINUX_COMP

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int hw842_available();

int hw842_compress(const uint8_t *in, size_t ilen,
		   uint8_t *out, size_t *olen);

int hw842_decompress(const uint8_t *in, size_t ilen,
		     uint8_t *out, size_t *olen);

int hw842_compress_chunked(size_t numchunks, int *rets,
			   const uint8_t *in, size_t istride, const size_t *ilens,
			   uint8_t *out, size_t ostride, size_t *olens);

int hw842_decompress_chunked(size_t numchunks, int *rets,
			     const uint8_t *in, size_t istride, const size_t *ilens,
			     uint8_t *out, size_t ostride, size_t *olens);

const struct lib842_implementation *get_hw842_implementation();

#ifdef __cplusplus
}
#endif

#endif

#endif // LIB842_HW_H
