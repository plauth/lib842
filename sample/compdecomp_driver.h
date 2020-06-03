#ifndef LIB842_SAMPLE_COMPDECOMP_DRIVER_H
#define LIB842_SAMPLE_COMPDECOMP_DRIVER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Those functions are provided by the driver
/**
 * Returns a timestamp suitable for benchmarking, in milliseconds.
 */
long long timestamp();

int compdecomp(const char *file_name, size_t chunk_size, size_t alignment);

// Those functions must be implemented by the driven code
bool simple_test_core(const uint8_t *in, size_t ilen,
		      uint8_t *out, size_t *olen,
		      uint8_t *decompressed, size_t *dlen);

bool compress_benchmark_core(const uint8_t *in, size_t ilen,
			     size_t *olen,
			     long long *time_comp,
			     long long *time_condense,
			     long long *time_decomp);

#ifdef __cplusplus
}
#endif

#endif // LIB842_SAMPLE_COMPDECOMP_DRIVER_H
