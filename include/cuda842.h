#ifndef __CUDA842_H__
#define __CUDA842_H__

#include <stdint.h>

#ifndef CUDA842_CHUNK_SIZE
#define CUDA842_CHUNK_SIZE 1024
#endif

#define CUDA842_STRICT

__global__ void cuda842_decompress(uint64_t *in, uint64_t *out);

#endif
