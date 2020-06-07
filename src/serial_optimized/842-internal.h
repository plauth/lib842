#ifndef LIB842_SRC_SERIAL_OPTIMIZED_842_INTERNAL_H
#define LIB842_SRC_SERIAL_OPTIMIZED_842_INTERNAL_H

#include "../common/842.h"

#define LIB842_CUDA_STRICT

// If enabled, enables handling of invalid bitstreams and undersized
// input or output buffers, which are exceptional errors that can be
// avoided in a controlled environment. This has a performance penalty.
#define ENABLE_ERROR_HANDLING

#define BRANCH_FREE (0)
//#define DEBUG 1

#endif // LIB842_SRC_SERIAL_OPTIMIZED_842_INTERNAL_H
