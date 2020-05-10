#ifndef LIB842_SRC_SERIAL_OPTIMIZED_842_INTERNAL_H
#define LIB842_SRC_SERIAL_OPTIMIZED_842_INTERNAL_H

#include "../common/842.h"
#include <cstdint>

#define LIB842_CUDA_STRICT

// If enabled, enables handling of invalid bitstreams and undersized
// input or output buffers, which are exceptional errors that can be
// avoided in a controlled environment. This has a performance penalty.
#define ENABLE_ERROR_HANDLING

/* If defined, avoid (ab)using undefined behaviour (as defined by the standard),
 * which nevertheless works on our target platforms and provides better performance.
 * This option is also useful to avoid warnings for debugging (e.g. valgrind). */
#define ONLY_WELL_DEFINED_BEHAVIOUR

#ifdef ENABLE_ERROR_HANDLING
#include <cstddef>
#endif

#define BRANCH_FREE (0)
//#define DEBUG 1

#define DICT16_BITS (10)
#define DICT32_BITS (11)
#define DICT64_BITS (10)

struct sw842_param {
	struct bitstream *stream;

	const uint8_t *in;
	const uint8_t *instart;
	uint64_t ilen;
	uint64_t olen;

	// 0-6: data; 7-13: indices; 14: 0
	uint64_t dataAndIndices[16];
	uint64_t hashes[7];
	uint16_t validity[7];
	uint16_t templateKeys[7];

	// L1D cache consumption: ~12.5 KiB
	int16_t hashTable16[1 << DICT16_BITS]; // 1024 * 2 bytes =   2 KiB
	int16_t hashTable32[1 << DICT32_BITS]; // 2048 * 2 bytes =   4 KiB
	int16_t hashTable64[1 << DICT64_BITS]; // 1024 * 2 bytes =   2 KiB
	uint16_t rollingFifo16[1 << I2_BITS]; // 256  * 2 bytes = 0.5 KiB
	uint32_t rollingFifo32[1 << I4_BITS]; // 512  * 4 bytes =   2 KiB
	uint64_t rollingFifo64[1 << I8_BITS]; // 256  * 8 bytes =   2 KiB
};

struct sw842_param_decomp {
	uint8_t *out;
	const uint8_t *ostart;
	const uint64_t *in;
#ifdef ENABLE_ERROR_HANDLING
	const uint64_t *istart;
	size_t ilen;
	size_t olen;
	int errorcode;
#endif
	uint8_t bits;
	uint64_t buffer;
};

#endif // LIB842_SRC_SERIAL_OPTIMIZED_842_INTERNAL_H
