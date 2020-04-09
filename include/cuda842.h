#ifndef __CUDA842_H__
#define __CUDA842_H__

#include "config842.h"

#ifdef LIB842_HAVE_CUDA

#include <stdint.h>

#define CUDA842_CHUNK_SIZE 1024
#define CUDA842_STRICT

__global__ void cuda842_decompress(const uint64_t *in, uint64_t *out);

#endif

#endif
