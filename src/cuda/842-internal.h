#ifndef LIB842_SRC_CUDA_842_INTERNAL_H
#define LIB842_SRC_CUDA_842_INTERNAL_H

#include "../common/842.h"
#include <lib842/cuda.h>

#ifndef LIB842_CUDA_STRICT
#undef I2_BITS
#undef I4_BITS
#undef I8_BITS
#define STATIC_LOG2_ARG LIB842_CUDA_CHUNK_SIZE
#include "../common/static_log2.h"
#define I2_BITS (STATIC_LOG2_VALUE - 1)
#define I4_BITS (STATIC_LOG2_VALUE - 2)
#define I8_BITS (STATIC_LOG2_VALUE - 3)
#endif

#endif // LIB842_SRC_CUDA_842_INTERNAL_H
