#ifndef LIB842_CUDA_H
#define LIB842_CUDA_H

#include <lib842/config.h>

#ifdef LIB842_HAVE_CUDA

#include <stdint.h>

#define LIB842_CUDA_CHUNK_SIZE 1024
#define LIB842_CUDA_STRICT

__global__ void cuda842_decompress(__restrict__ const uint64_t *in,
				   __restrict__ uint64_t *out);

#endif

#endif // LIB842_CUDA_H
