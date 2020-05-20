#ifndef LIB842_SRC_SERIAL_OPTIMIZED_842_INTERNAL_H
#define LIB842_SRC_SERIAL_OPTIMIZED_842_INTERNAL_H

#include "../common/842.h"

#define LIB842_CUDA_STRICT

// If enabled, enables handling of invalid bitstreams and undersized
// input or output buffers, which are exceptional errors that can be
// avoided in a controlled environment. This has a performance penalty.
#define ENABLE_ERROR_HANDLING

/* If defined, avoid (ab)using undefined behaviour (as defined by the standard),
 * which nevertheless works on our target platforms and provides better performance.
 * This option is also useful to avoid warnings for debugging (e.g. valgrind). */
#define ONLY_WELL_DEFINED_BEHAVIOUR

#define BRANCH_FREE (0)
//#define DEBUG 1

#endif // LIB842_SRC_SERIAL_OPTIMIZED_842_INTERNAL_H
