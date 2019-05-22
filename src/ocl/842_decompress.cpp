#include "clutil.hpp"
#include "cl842kernels.hpp"
#include <sys/time.h>
#include <iostream>

#include "842-internal.h"

using namespace std;

#define STRLEN 32

long long timestamp() {
    struct timeval te;
    gettimeofday(&te, NULL);
    long long ms = te.tv_sec * 1000LL + te.tv_usec/1000;
    return ms;
}

int nextMultipleOfChunkSize(unsigned int input) {
    unsigned int size = CHUNK_SIZE * THREADS_PER_BLOCK;
    return (input + (size-1)) & ~(size-1);
} 

int main(int argc, char *argv[]) {
    CL842Kernels kernels;

    uint8_t *inH, *compressedH, *decompressedH;
    cl::Buffer compressedD, decompressedD;

    inH = compressedH = decompressedH = NULL;

    uint32_t ilen, olen, dlen;
    ilen = olen = dlen = 0;
    int64_t timestart_comp, timeend_comp;
    int64_t timestart_decomp, timeend_decomp;

    if(argc <= 1) {
        ilen = STRLEN;
        olen = ilen * 2;
        dlen = ilen;

        inH = (uint8_t*) malloc(ilen);
        compressedH = (uint8_t*) malloc(olen);
        decompressedH = (uint8_t*) malloc(dlen);
        memset(inH, 0, ilen);
        memset(compressedH, 0, olen);
        memset(decompressedH, 0, dlen);

        compressedD     = kernels.allocateBuffer(olen, CL_MEM_READ_ONLY);
        decompressedD   = kernels.allocateBuffer(dlen, CL_MEM_READ_WRITE);
        kernels.fillBuffer(compressedD, 0, 0, olen);
        kernels.fillBuffer(decompressedD, 0, 0, dlen);

        uint8_t tmp[] = {0x30, 0x30, 0x31, 0x31, 0x32, 0x32, 0x33, 0x33, 0x34, 0x34, 0x35, 0x35, 0x36, 0x36, 0x37, 0x37, 0x38, 0x38, 0x39, 0x39, 0x40, 0x40, 0x41, 0x41, 0x42, 0x42, 0x43, 0x43, 0x44, 0x44, 0x45, 0x45};//"0011223344556677889900AABBCCDDEE";
        strncpy((char *) inH, (const char *) tmp, STRLEN);
    }  else if (argc == 2) {
        FILE *fp;
        fp=fopen(argv[1], "r");
        fseek(fp, 0, SEEK_END);
        unsigned int flen = ftell(fp);
        ilen = flen;
        printf("original file length: %d\n", ilen);
        ilen = nextMultipleOfChunkSize(ilen);
        printf("original file length (padded): %d\n", ilen);
        olen = ilen * 2;
        dlen = ilen;
        fseek(fp, 0, SEEK_SET);

        inH = (uint8_t*) malloc(ilen);
        compressedH = (uint8_t*) malloc(olen);
        decompressedH = (uint8_t*) malloc(dlen);
        memset(inH, 0, ilen);
        memset(compressedH, 0, olen);
        memset(decompressedH, 0, dlen);

        compressedD     = kernels.allocateBuffer(olen, CL_MEM_READ_ONLY);
        decompressedD   = kernels.allocateBuffer(dlen, CL_MEM_READ_WRITE);
        kernels.fillBuffer(compressedD, 0, 0, olen);
        kernels.fillBuffer(decompressedD, 0, 0, dlen);


        if(!fread(inH, flen, 1, fp)) {
            fprintf(stderr, "FAIL: Reading file content to memory failed.\n");
        }
        fclose(fp);
    }

    compressedD     = kernels.allocateBuffer(olen, CL_MEM_READ_ONLY);
    decompressedD   = kernels.allocateBuffer(dlen, CL_MEM_READ_WRITE);



    if(ilen > CHUNK_SIZE) {
        printf("Using chunks of %d bytes\n", CHUNK_SIZE);

        uint32_t num_chunks = ilen / CHUNK_SIZE;
    
        timestart_comp = timestamp();
        #pragma omp parallel for
        for(uint32_t chunk_num = 0; chunk_num < num_chunks; chunk_num++) { 
            uint32_t chunk_olen = CHUNK_SIZE * 2;
            uint8_t* chunk_in = inH + (CHUNK_SIZE * chunk_num);
            uint8_t* chunk_out = compressedH + ((CHUNK_SIZE * 2) * chunk_num);
            
            sw842_compress(chunk_in, CHUNK_SIZE, chunk_out, &chunk_olen);
        }

        timeend_comp = timestamp();

        printf("Threads per Block: %d\n", THREADS_PER_BLOCK );

        kernels.writeBuffer(compressedD, (const void*) compressedH, olen);

        timestart_decomp = timestamp();

        kernels.decompress(compressedD, decompressedD, num_chunks);

        timeend_decomp = timestamp();

        kernels.readBuffer(decompressedD, (void*) decompressedH, dlen);


        printf("Compression performance: %lld ms / %f MiB/s\n", timeend_comp - timestart_comp, (ilen / 1024 / 1024) / ((float) (timeend_comp - timestart_comp) / 1000));
        printf("Decompression performance: %lld ms / %f MiB/s\n", timeend_decomp - timestart_decomp, (ilen / 1024 / 1024) / ((float) (timeend_decomp - timestart_decomp) / 1000));


    } else {

        sw842_compress(inH, ilen, compressedH, &olen);

        kernels.writeBuffer(compressedD, (const void*) compressedH, olen);

        kernels.decompress(compressedD, decompressedD, 1);
        
        kernels.readBuffer(decompressedD, (void*) decompressedH, dlen);
    }
    

    if (memcmp(inH, decompressedH, ilen) == 0) {
        printf("Compression- and decompression was successful!\n");
    } else {

        for (uint32_t i = 0; i < olen; i++) {
            printf("%02x:", compressedH[i]);
        }

        printf("\n\n");

        for (uint32_t i = 0; i < dlen; i++) {
            printf("%02x:", decompressedH[i]);
        }

        printf("\n\n");

        fprintf(stderr, "FAIL: Decompressed data differs from the original input data.\n");
    }

    return 0;
    
}
