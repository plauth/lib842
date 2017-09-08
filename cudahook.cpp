#include <dlfcn.h>
#include <stdio.h>
#include "cpu_serial/842.h"
#include <cassert>

#include <cuda_runtime_api.h>

int nextMultipleOfEight(unsigned int input) {
	return (input + 7) & ~7;
} 

extern __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
	__host__ cudaError_t CUDARTAPI (*original_cudaMemcpy)(void*, const void*, size_t, cudaMemcpyKind);
	original_cudaMemcpy = (cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind)) dlsym(RTLD_NEXT, __func__);
        if(kind == cudaMemcpyHostToDevice) {
                void* wmem_comp = malloc(sizeof(struct sw842_param)); 
                uint8_t *in, *out, *decompressed;
                unsigned int ilen, olen, dlen;
                ilen = nextMultipleOfEight(count);
                olen = ilen * 2;
                dlen = ilen;
                
                in = (uint8_t*) malloc(ilen);
                out = (uint8_t*) malloc(olen);
                decompressed = (uint8_t*) malloc(dlen);

                memset(in, 0, ilen);
                memset(out, 0, olen);
                memset(decompressed, 0, dlen);

                memcpy(in, src, count);

                sw842_compress(in, ilen, out, &olen, wmem_comp);
                
                printf("\n");
                printf("### lib842 test-hook (start) ###\n");
                printf("Input: %d bytes (padded)\n",ilen);
                printf("Output: %d bytes\n", olen);

                printf("Compression factor: %f\n", (float) olen / (float) ilen);
                sw842_decompress(out, olen, decompressed, &dlen);
        
                if (memcmp(in, decompressed, ilen) == 0) {
                        //printf("Compression- and decompression was successful!\n");
                } else {
                        fprintf(stderr, "FAIL: Decompressed data differs from the original input data.\n");
                }
                printf("### lib842 test-hook (end) ###\n");
        }
	return (*original_cudaMemcpy)(dst, src, count, kind);
}
