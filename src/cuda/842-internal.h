
#ifndef __842_INTERNAL_H__
#define __842_INTERNAL_H__

#include "../common/842.h"
#include "../../include/cuda842.h"

#ifndef CUDA842_STRICT
#undef I2_BITS
#undef I4_BITS
#undef I8_BITS
#define STATIC_LOG2_ARG CUDA842_CHUNK_SIZE
#include "../common/static_log2.h"
#define I2_BITS (STATIC_LOG2_VALUE - 1)
#define I4_BITS (STATIC_LOG2_VALUE - 2)
#define I8_BITS (STATIC_LOG2_VALUE - 3)
#endif

struct sw842_param_decomp {
	uint64_t *out;
	const uint64_t *ostart;
	const uint64_t *in;
	uint32_t bits;
	uint64_t buffer;
};

#endif
