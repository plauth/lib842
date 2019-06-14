#include "clutil.hpp"
#include "cl842kernels.hpp"
#include <iostream>
#include <sys/time.h>

#include "../../include/sw842.h"

using namespace std;

#define STRLEN 32

long long timestamp() {
    struct timeval te;
    gettimeofday(&te, NULL);
    long long ms = te.tv_sec * 1000LL + te.tv_usec/1000;
    return ms;
}

int main(int argc, char *argv[]) {
    uint8_t *compressedH, *decompressedH;
    uint32_t olen, dlen, num_chunks;
    long long timestart_decomp, timeend_decomp;


    fread(&olen, sizeof(uint32_t), 1, stdin);
    fread(&dlen, sizeof(uint32_t), 1, stdin);
    fread(&num_chunks, sizeof(uint32_t), 1, stdin);

    fprintf(stderr, "(decompress): olen = %d, dlen = %d, num_chunks = %d\n", olen, dlen, num_chunks);

    compressedH = (uint8_t*) malloc(olen);
    decompressedH = (uint8_t*) malloc(dlen);

    fread(compressedH, sizeof(uint8_t), olen, stdin);


    CL842Kernels kernels;
    
    cl::Buffer compressedD     = kernels.allocateBuffer(olen, CL_MEM_READ_ONLY);
    cl::Buffer decompressedD   = kernels.allocateBuffer(dlen, CL_MEM_READ_WRITE);
    kernels.fillBuffer(compressedD, 0, 0, olen);
    kernels.fillBuffer(decompressedD, 0, 0, olen);

    if(num_chunks > 1) {
        fprintf(stderr, "Threads per Block: %d\n", THREADS_PER_BLOCK );
        kernels.writeBuffer(compressedD, (const void*) compressedH, olen);
        timestart_decomp = timestamp();
        kernels.decompress(compressedD, decompressedD, num_chunks);
        timeend_decomp = timestamp();
        kernels.readBuffer(decompressedD, (void*) decompressedH, dlen);
    } else {
        kernels.writeBuffer(compressedD, (const void*) compressedH, olen);
        kernels.decompress(compressedD, decompressedD, 1);
        kernels.readBuffer(decompressedD, (void*) decompressedH, dlen);

    }
    

    fprintf(stderr, "Decompression performance: %lld ms / %f MiB/s\n", timeend_decomp - timestart_decomp, (dlen / 1024 / 1024) / ((float) (timeend_decomp - timestart_decomp) / 1000));
    
    fwrite(decompressedH, sizeof(uint8_t), dlen, stdout);

    return 0;
    
}
